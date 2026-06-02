import streamlit as st
import tempfile

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

# Title
st.title("HR Policy RAG Chatbot")

# API key from Streamlit secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

# Load Groq model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload HR Policy PDF",
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
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    # Lightweight embeddings
    embeddings = FakeEmbeddings(size=384)

    # Store in FAISS vector database
    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    # User question input
    user_question = st.text_input(
        "Ask your question"
    )

    if user_question:

        # Similarity search
        relevant_docs = vectorstore.similarity_search(
            user_question
        )

        # Combine retrieved text
        context = "\n".join(
            [doc.page_content for doc in relevant_docs]
        )

        # Final prompt
        final_prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{user_question}
"""

        # LLM response
        response = llm.invoke(final_prompt)

        st.write(response.content)
