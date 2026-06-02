import streamlit as st
import tempfile

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Title
st.title("Simple RAG Chatbot")

# API Key
groq_api_key = st.secrets["GROQ_API_KEY"]

# Load LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file is not None:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("PDF uploaded successfully")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    st.write("PDF Pages Loaded:", len(docs))

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = text_splitter.split_documents(docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector database
    vectorstore = FAISS.from_documents(
        split_docs,
        embeddings
    )

    # User question
    question = st.text_input("Ask Question")

    if question:

        # Similarity Search
        relevant_docs = vectorstore.similarity_search(
            question,
            k=5
        )

        # Combine retrieved chunks
        context = "\n".join(
            [doc.page_content for doc in relevant_docs]
        )

        # Final Prompt
        prompt = f"""
Answer the question using the PDF content below.

Context:
{context}

Question:
{question}
"""

        # LLM Response
        response = llm.invoke(prompt)

        st.write(response.content)
