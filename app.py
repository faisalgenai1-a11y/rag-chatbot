import streamlit as st
import tempfile

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader

st.title("Simple RAG Chatbot")

groq_api_key = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("PDF uploaded successfully")

    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    st.write("PDF Pages Loaded:", len(docs))

    question = st.text_input("Ask Question")

    if question:

        context = docs[0].page_content

        prompt = f"""
Answer using this PDF content:

{context}

Question:
{question}
"""

        response = llm.invoke(prompt)

        st.write(response.content)
