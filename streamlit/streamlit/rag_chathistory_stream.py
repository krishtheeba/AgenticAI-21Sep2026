import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage

# STREAMLIT UI SETUP
# 
st.set_page_config(page_title="RAG Chatbot with History", layout="centered")
st.title("RAG Chatbot with Chat History")

# INITIALIZE SESSION STATE 

if "rag_chain" not in st.session_state:
    
    # Step-1 Load PDF
    loader = PyPDFLoader("attention.pdf")
    documents = loader.load()

    # Step-2 Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Step-3 Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Step-4 Vectorstore
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever_obj = vector_store.as_retriever()

    # Step-5 LLM object
    llm_obj = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Step-6 Prompt
    prompt = ChatPromptTemplate.from_template("""
Use the following context and chat history to answer the question.

Chat history:
{history}

Context:
{context}

Question:
{input}
""")

    # Step-7 Build RAG chain
    qa_chain = create_stuff_documents_chain(llm_obj, prompt)
    rag_chain = create_retrieval_chain(retriever_obj, qa_chain)

    def convert_output_to_aimessage(output):
        return AIMessage(content=output["answer"])

    final_rag_chain = rag_chain | convert_output_to_aimessage

    # Step-8 Add message history
    stores = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in stores:
            stores[session_id] = ChatMessageHistory()
        return stores[session_id]

    rag_with_history = RunnableWithMessageHistory(
        final_rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    st.session_state.rag_chain = rag_with_history


# CHAT INTERFACE

session_id = "streamlit-session-1"

# Create chat history UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# User query text box
query = st.chat_input("Ask something...")

if query:
    # Show in chat
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Run RAG pipeline
    response = st.session_state.rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}}
    )

    ai_text = response.content

    # Display AI message
    st.session_state.messages.append({"role": "assistant", "content": ai_text})

    with st.chat_message("assistant"):
        st.write(ai_text)
