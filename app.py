import streamlit as st
import os
import re
import pandas as pd
from new import process_pdf, process_csv, query_and_respond, generate_sql_and_query_db

# === Config ===
groq_api_key = st.secrets["GROQ_API_KEY"]
gemini_api_key = st.secrets["GEMINI_API_KEY"]

# === PostgreSQL DB Config ===
db_config = {
    "host": "localhost",
    "database": "test",
    "user": "postgres",
    "password": "mysql",
    "port": "5432"
}

st.set_page_config(page_title="📄 PDF, CSV & DB Q&A", layout="wide")
st.title("📄 PDF, CSV & Database Question Answering App")

# === Top-Level Mode Switch ===
query_mode = st.radio("📌 What do you want to query?", ["PDF", "CSV", "Database (PostgreSQL)"])
uploaded_file = None

# === Conditional File Upload ===
if query_mode in ["PDF", "CSV"]:
    file_types = []
    if query_mode == "PDF":
        file_types = ["pdf"]
    elif query_mode == "CSV":
        file_types = ["csv"]
    else:
        file_types = ["pdf", "csv"]
    uploaded_file = st.file_uploader("📂 Upload your document", type=file_types)

# === File Processing ===
if uploaded_file and query_mode in ["PDF", "CSV",]:
    ext = uploaded_file.name.split(".")[-1].lower()
    os.makedirs("temp", exist_ok=True)
    file_path = os.path.join("temp", uploaded_file.name)
    persist_path = os.path.join("chroma_store", uploaded_file.name.replace(f".{ext}", ""))

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Preview CSV
    if ext == "csv":
        try:
            df = pd.read_csv(file_path)
            st.subheader("📊 CSV Preview:")
            st.dataframe(df.head())
        except Exception as e:
            st.warning(f"Could not preview CSV: {e}")

    # Reset old vectordb
    if "last_file" in st.session_state and st.session_state["last_file"] != uploaded_file.name:
        st.session_state.pop("vectordb", None)
    st.session_state["last_file"] = uploaded_file.name

    if st.button("🚀 Process Document"):
        with st.spinner("Extracting and indexing..."):
            if ext == "pdf":
                vectordb = process_pdf(file_path, persist_path)
            elif ext == "csv":
                vectordb = process_csv(file_path, persist_path)
            else:
                st.error("Unsupported file type.")
                vectordb = None

            if vectordb:
                st.session_state["vectordb"] = vectordb
                st.success("✅ Document indexed. Ask away!")

# === Helper for Formatting ===
def format_answer(text):
    if ':' in text:
        parts = re.split(r'(?<=:),? ?', text)
        grouped = []
        for i in range(0, len(parts), 2):
            pair = ":".join(parts[i:i+2]).strip()
            if pair:
                grouped.append(pair)
        return grouped
    else:
        sentences = re.split(r'(?<=[.!?]) +', text)
        return [" ".join(sentences[i:i+2]).strip() for i in range(0, len(sentences), 2)]

# === Clear input field ===
def clear_input():
    st.session_state.query_input = ""

# === Question Section ===
if query_mode == "Database (PostgreSQL)" or "vectordb" in st.session_state:
    st.subheader("💬 Ask a question:")
    if query_mode != "Database (PostgreSQL)":
        lang = st.selectbox("🌐 Select language:", ["English", "Gujarati"])
    else:
        lang = "English"  # Default fallback

    model_choice = st.radio("🤖 Choose model:", ["LLaMA", "Gemini 2.0 Flash"])


    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Your question:", key="query_input")

    with col2:
        if st.button("❌ Clear", key="clear_button"):
            st.session_state.query_input = ""

    if query:
        with st.spinner("🧠 Thinking..."):
            answers = []

            if query_mode in ["PDF", "CSV"]:
                answers.extend(query_and_respond(
                    st.session_state["vectordb"],
                    query,
                    groq_api_key,
                    gemini_api_key,
                    model_choice
                ))

            if query_mode in ["Database (PostgreSQL)", "Both"]:
                answers.extend(generate_sql_and_query_db(query, db_config, groq_api_key))

            for ans in answers:
                key_values = [kv.strip() for kv in ans.split(",")]
                st.markdown("🔹 " + " | ".join(key_values))
