import os
import re
import requests
import shutil
import pdfplumber
import pandas as pd
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import google.generativeai as genai
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL Database Configuration
db_config = {
    "host": "localhost",
    "database": "test",          # Replace with your database name
    "user": "postgres",             #  Or your PostgreSQL username
    "password": "mysql",    #  Replace with your password
    "port": "5432"
}

# === Clean Text ===
def clean_text(text):
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

# === Smart Table Extraction for Matrix Format (Entity in Columns) ===
def extract_table_chunks(pdf_path):
    table_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2 or len(table[0]) < 2:
                    continue

                row_dict = {}
                for row in table:
                    key = row[0].strip() if row[0] else ""
                    values = [cell.strip() for cell in row[1:] if cell]
                    if key:
                        row_dict[key] = values

                num_entities = max(len(v) for v in row_dict.values())

                for idx in range(num_entities):
                    record = {}
                    for key, val_list in row_dict.items():
                        record[key] = val_list[idx] if idx < len(val_list) else ""

                    entity_name = record.get("Crop", record.get("Entity", f"record_{idx}"))
                    import json
                    chunk_json = json.dumps(record, ensure_ascii=False)
                    table_chunks.append({
                        "page_content": chunk_json,
                        "metadata": {
                            "type": "table",
                            "page": i + 1,
                            "entity": entity_name.strip().lower(),
                            "row_id": f"page_{i+1}col{idx + 1}"
                        }
                    })
    return table_chunks

# === Process PDF ===
def process_pdf(pdf_path, persist_path):
    if os.path.exists(persist_path):
        shutil.rmtree(persist_path)
    os.makedirs(persist_path, exist_ok=True)

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_chunks = [{
        "page_content": clean_text(doc.page_content),
        "metadata": {"type": "text", "page": i + 1}
    } for i, doc in enumerate(documents)]

    table_chunks = extract_table_chunks(pdf_path)
    all_docs = text_chunks + table_chunks

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = [doc["page_content"] for doc in all_docs]
    metas = [doc["metadata"] for doc in all_docs]
    chunks = splitter.create_documents(texts, metadatas=metas)

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb = Chroma(
        collection_name="rag_text_table_demo",
        embedding_function=embedder,
        persist_directory=persist_path
    )

    vectordb.add_documents(chunks)
    return vectordb

# === Process CSV ===
def process_csv(csv_path, persist_path):
    if os.path.exists(persist_path):
        shutil.rmtree(persist_path)
    os.makedirs(persist_path, exist_ok=True)

    df = pd.read_csv(csv_path)
    docs = []
    for i, row in df.iterrows():
        text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        docs.append(Document(page_content=text, metadata={"row": i + 1, "type": "csv"}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        collection_name="rag_text_table_demo",
        embedding_function=embedder,
        persist_directory=persist_path
    )

    vectordb.add_documents(chunks)
    return vectordb

# === Extract Subjects from Question (Multiple Terms) ===
def extract_subjects(question):
    match = re.search(r"(?:for|of|about|regarding)\s+([a-zA-Z0-9,\s]+)", question.lower())
    subject_str = match.group(1).strip() if match else None
    subjects = [s.strip() for s in subject_str.split(',')] if subject_str else []
    return subjects

# === Query and Generate Answer ===
def detect_language(text):
        # Basic check for Gujarati Unicode range
        for char in text:
            if '\u0A80' <= char <= '\u0AFF':
                return "Gujarati"
        return "English"

def query_and_respond(vectordb, query, groq_api_key, gemini_api_key, model_choice):
    subjects = extract_subjects(query)
    results = vectordb.similarity_search_with_score(query, k=5)

    if subjects:
        results = [
            (doc, score) for doc, score in results
            if any(f"*entity*: {sub.lower()}" in doc.page_content.lower() for sub in subjects)
        ]

    if not results:
        results = vectordb.similarity_search_with_score(query, k=12)

    seen = set()
    context_parts = []
    for doc, _ in results:
        clean = doc.page_content.strip()
        if clean not in seen:
            seen.add(clean)
            context_parts.append(clean)

    context = "\n\n---\n\n".join(context_parts)

    lang = detect_language(query)

    if lang == "Gujarati":
        instruction = "ઉત્તર ગુજરાતી ભાષામાં આપો. જો જવાબ ઉપલબ્ધ ન હોય, તો 'મને ખબર નથી' કહો."
    else:
        instruction = "Respond in English. If the answer is not available, say 'I don't know.'"

    # 🧠 Prompt to send to the model
    prompt = f"""You are a helpful assistant. Read the instruction carefully and do as it says. Use ONLY the context below to answer the user's question.
    {instruction}

    Context:
    {context}

    Question: {query}
    Answer:"""

    if "llama" in model_choice.lower() or "groq" in model_choice.lower():
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You answer only based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,
            "max_tokens": 2048,
            "top_p": 0.1,
            "frequency_penalty": 1,
            "presence_penalty": 0
        }

        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"].strip()
            return [answer] if answer else ["I don't know."]
        else:
            return [f"Error: {response.status_code} - {response.text}"]

    elif model_choice.lower().startswith("gemini"):
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("models/gemini-2.0-flash")
            response = model.generate_content(prompt)
            return [response.text.strip()]
        except Exception as e:
            return [f"Gemini error: {e}"]

    return ["Model not supported."]


def generate_sql_and_query_db(question, db_config, groq_api_key):
    try:
        # Step 1: Connect and extract schema and relationships
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        # Get table + column info
        cur.execute("""
            SELECT table_name, column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'public';
        """)
        columns_metadata = cur.fetchall()

        # Get foreign key relationships
        cur.execute("""
            SELECT 
                tc.table_name AS source_table,
                kcu.column_name AS source_column,
                ccu.table_name AS target_table,
                ccu.column_name AS target_column
            FROM 
                information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY';
        """)
        fk_metadata = cur.fetchall()
        cur.close()

        # Step 2: Build schema string
        schema = {}
        for table, column in columns_metadata:
            schema.setdefault(table, []).append(column)

        schema_str = "\n".join([f"Table {table} → Columns: {', '.join(cols)}" for table, cols in schema.items()])
        #print("📘 Schema passed to model:")
        #print(schema_str)

        fk_str = "\n".join([
            f"{src}.{src_col} → {tgt}.{tgt_col}"
            for src, src_col, tgt, tgt_col in fk_metadata
        ])

        # Step 3: Build prompt
        prompt = f"""
You are a PostgreSQL SQL assistant. Convert the user's question into a valid SELECT-only SQL query using the schema below.
Always start with a FROM clause. Use  JOINs correctly to connect related tables. 
Use LOWER(column) = 'value' or ILIKE for filtering text fields to avoid case sensitivity issues.
Only output the SQL query — no explanation or heading.
 
📌 Tables and Columns:
{schema_str}

🔗 Foreign Key Relationships:
{fk_str or 'None'}

User Question:
{question}

Only return the final SQL query and nothing else.
"""

        # Step 4: Send to LLM
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You are a SQL query generator."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 256
        }

        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        # Step 4.5: Clean SQL and display
        raw_sql = response.json()["choices"][0]["message"]["content"]

        # Strip code block markers
        raw_sql = raw_sql.strip().strip("```sql").strip("```")
        # Handle gender and crop_type case-insensitive filtering even if prefixed
        raw_sql = re.sub(r"(?i)(\w+\.)?gender\s*=\s*'male'", lambda m: f"LOWER({m.group(1) or ''}gender) = 'male'", raw_sql)
        raw_sql = re.sub(r"(?i)(\w+\.)?gender\s*=\s*'female'", lambda m: f"LOWER({m.group(1) or ''}gender) = 'female'", raw_sql)
        raw_sql = re.sub(r"(?i)(\w+\.)?crop_type\s*=\s*'(\w+)'", lambda m: f"LOWER({m.group(1) or ''}crop_type) = '{m.group(2).lower()}'", raw_sql)
        raw_sql = re.sub(r"(?i)gender\s*=\s*'female'", "LOWER(gender) = 'female'", raw_sql)
        raw_sql = re.sub(r"(?i)crop_type\s*=\s*'(\w+)'", lambda m: f"LOWER(crop_type) = '{m.group(1).lower()}'", raw_sql)

        # Remove introductory lines like "Here is the SQL:"
        sql_lines = raw_sql.splitlines()
        sql_lines = [line for line in sql_lines if not re.match(r"^(here|sql|query)", line.strip().lower())]
        sql = "\n".join(sql_lines).strip("; \n")

        # Debug: Show generated SQL before execution
        print("🔍 Generated SQL:")
        print(sql)

        # Step 5: Run query
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        # Optionally allow the user to preview and confirm
        #confirm = input("Execute this SQL? (y/n): ").strip().lower()
        #if confirm != 'y':
         #   print("⛔ Query skipped by user.")
          #  return ["Query skipped."]
        cur.execute(sql)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()

        results = []
        for row in rows:
            results.append(", ".join(f"{col}: {val}" for col, val in zip(columns, row)))
        return results or ["✅ SQL executed but returned no rows."]

    except Exception as e:
        return [f"⚠️ SQL Error:\n{str(e)}"]

#MAIN RUNNER 
def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    persist_path = "./chroma_store"

    pdf_path = input("Enter PDF path: ").strip()
    if not os.path.exists(pdf_path):
        print("File not found.")
        return

    vectordb = process_pdf(pdf_path, persist_path)

    while True:
        user_query = input("\n Ask your question (type 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            break
            # Try SQL query first
        sql_results = generate_sql_and_query_db(user_query, db_config, groq_api_key)
        if sql_results and isinstance(sql_results, list) and not str(sql_results[0]).startswith("❌"):
            print("🟢 SQL Result:")
            for row in sql_results:
                print(row)
            continue  # Skip PDF/CSV RAG if SQL result was found

    query_and_respond(vectordb, user_query, groq_api_key)

if __name__ == "__main__":
    main()
