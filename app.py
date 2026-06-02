import streamlit as st
from langchain_groq import ChatGroq

st.title("RAG AI Chatbot")

groq_api_key = st.secrets["GROQ_API_KEY"]

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

user_question = st.text_input("Ask your question")

if user_question:
    response = llm.invoke(user_question)
    st.write(response.content)
