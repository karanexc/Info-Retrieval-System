import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load .env for local development
load_dotenv()

# Use Streamlit secrets first, fallback to .env
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# ------------------ PDF Text Extraction ------------------

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# ------------------ Text Chunking ------------------

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return text_splitter.split_text(text)

# ------------------ Vector Store (FAISS) ------------------

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        disallowed_special=()  # avoid tokenization issues
    )
    return FAISS.from_texts(text_chunks, embedding=embeddings)

# ------------------ Conversational Chain ------------------

def get_conversational_chain(vector_store):
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
