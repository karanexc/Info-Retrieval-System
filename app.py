import streamlit as st
import os
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.ChatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.ChatHistory):
        if i % 2 == 0:
            st.write("User: ", message.content)
        else:
            st.write("Reply: ", message.content)

def main():
    st.set_page_config(page_title="Information Retrieval System", layout="wide")
    st.header("Information Retrieval System")

    user_question = st.text_input("Ask a question about the uploaded PDFs:")

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if "ChatHistory" not in st.session_state:
        st.session_state.ChatHistory = None
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and click on Submit Button", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing.."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("PDFs processed successfully!")





if __name__ == "__main__":
    main()
    # Uncomment the line below to run the app with Streamlit
    # st.run()

