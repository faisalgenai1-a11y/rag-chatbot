import streamlit as st
import tempfile

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Streamlit title
st.title("HR Policy RAG Chatbot")

# Load Groq API Key
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
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("PDF uploaded successfully")

    # Load PDF
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    st.write("Total Pages:", len(documents))

    # Text Splitter (same as notebook)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "]
    )

    chunks = text_splitter.split_documents(documents)

    st.write("Total Chunks:", len(chunks))

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # FAISS Vector Store
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    # User Question
    user_question = st.text_input(
        "Ask your question"
    )

    if user_question:

        # Retrieve relevant chunks
        relevant_docs = retriever.invoke(user_question)

        # Combine context
        context = "\n\n".join(
            [doc.page_content for doc in relevant_docs]
        )

        # Final Prompt
        final_prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided PDF context.

If answer is not available in context,
say:
"I could not find this information in the PDF."

Context:
{context}

Question:
{user_question}
"""

        # Generate response
        response = llm.invoke(final_prompt)

        # Show answer
        st.write("### Answer")
        st.write(response.content)
