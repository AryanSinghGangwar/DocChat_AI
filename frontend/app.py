import streamlit as st
import requests
import os

# Configure the API URL (can be overridden via environment variable)
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="DocChat AI", page_icon="📄", layout="centered")

st.title("📄 DocChat AI")
st.markdown("Upload a PDF and ask questions about its content. Powered by Gemini 2.0 and Sentence Transformers.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for document upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Upload & Process"):
            with st.spinner("Uploading document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{API_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success("Document uploaded successfully!")
                        st.session_state.document_id = uploaded_file.name
                        # Clear chat history on new upload
                        st.session_state.messages = []
                    else:
                        st.error(f"Error uploading document: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
                    
    if "document_id" in st.session_state:
        st.success(f"Current Document: {st.session_state.document_id}")

# Main chat interface
st.subheader("Chat")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your document..."):
    if "document_id" not in st.session_state:
        st.error("Please upload a document first.")
    else:
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {
                        "document_id": st.session_state.document_id,
                        "query": prompt
                    }
                    response = requests.post(f"{API_URL}/chat", json=payload)
                    
                    if response.status_code == 200:
                        answer = response.json().get("answer", "No answer provided.")
                        st.markdown(answer)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
