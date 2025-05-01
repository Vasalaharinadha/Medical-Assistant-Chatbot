from dotenv import load_dotenv
import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import torch

# Load environment variables from .env file
load_dotenv()

# Retrieve HF_TOKEN from environment
HF_TOKEN = os.getenv("HF_TOKEN")

# Check if the token is retrieved
if not HF_TOKEN:
    print("HF_TOKEN is missing. Please set the Hugging Face token.")
else:
    print(f"HF_TOKEN loaded successfully: {HF_TOKEN[:10]}...")  # Print first 10 chars of token for verification

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Corrected load_llm function
def load_llm(huggingface_repo_id, HF_TOKEN):
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Check if GPU is available
    
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 512, "device": device}
    )
    return llm

def main():
    st.title("Medical Assistant Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer the user's question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            Don't provide anything out of the given context

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk please.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

        try:
            st.write("Loading vector store...")
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return
            st.write("Vector store loaded successfully.")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            st.write("Starting model inference...")
            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            # You can still keep source docs for logging or internal use
            source_documents = response.get("source_documents", [])

            # Only show the answer to the user, without source documents
            result_to_show = result  # No source docs in the UI

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
