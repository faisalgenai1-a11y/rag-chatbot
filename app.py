import streamlit as st
import tempfile

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Title
st.title("Simple RAG Chatbot")

# Groq API Key
groq_api_key = st.secrets["GROQ_API_KEY"]

# Load LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192"
)

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file is not None:

    # Save temp PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("PDF uploaded successfully")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    st.write("Total Pages:", len(documents))

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)

    st.write("Total Chunks:", len(docs))

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector DB
    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    # Question input
    question = st.text_input("Ask Question")

    if question:

        # Similarity search
        relevant_docs = vectorstore.similarity_search(
            question,
            k=4
        )

        # Combine context
        context = "\n\n".join(
            [doc.page_content for doc in relevant_docs]
        )

        # Prompt
        prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

        # LLM Response
        response = llm.invoke(prompt)

        st.write(response.content)
