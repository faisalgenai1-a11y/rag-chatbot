import streamlit as st
import tempfile
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Title
st.title("Simple RAG Chatbot")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Groq API Key
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

    # Create vector DB only first time
    if st.session_state.vectorstore is None:

        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        st.success("PDF uploaded successfully")

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        st.write("Total Pages:", len(documents))

        # Split into chunks
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

    # Create Vector Store only once
    if st.session_state.vectorstore is None:
    
        st.write("Creating Vector DB....")
    
        st.session_state.vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )
    
        st.write("Vector DB Ready")

    vectorstore = st.session_state.vectorstore

    # Show ALL old messages first
    for msg in st.session_state.messages:
        if msg["role"] == "User":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

    # Question Input
    question = st.chat_input("Ask a Question about your PDF")
    
    
    if question:
        st.write("Question receive")
        start_time = time.time()

        #Save and show user question immediately
        st.session_state.messages.append({
            "role": "User",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        # Search relevant chunks
        relevant_docs = vectorstore.similarity_search(
            question,
            k=3
        )

        # Combine context
        context = "\n\n".join(
            [doc.page_content for doc in relevant_docs]
        )

        # Prompt
        prompt = f"""
You are a helpful assistant.

Answer ONLY from the provided context.

If answer is not available in context,
say:
"I could not find this information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

        try:
            st.write("Calling Groq....")
            response = llm.invoke(prompt)
            st.write("Groq Returned answer")

            sources = []
            for doc in relevant_docs:
                if "page" in doc.metadata:
                    sources.append(
                        f"Page {doc.metadata['page'] + 1}"
                    )

            # Save assistant response
            st.session_state.messages.append({
                "role": "Assistant",
                "content": response.content
            })

            # Show answer in proper chat bubble
            with st.chat_message("assistant"):
                st.write(response.content)
                end_time = time.time()

                st.write(f"Response Time: {end_time -start_time: .2f} seconds")
                         
                if sources:
                    st.write("### Sources")
                    for source in sources:
                        st.write(source)

        except Exception as e:
            st.error(str(e))
