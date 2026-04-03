import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Ρυθμίσεις σελίδας για WordPress
st.set_page_config(page_title="AI Assistant", layout="centered")

# --- 1. Φόρτωση Μοντέλων & Βάσης ---
@st.cache_resource
def init_rag():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Φόρτωση της έτοιμης βάσης που ανέβασες στο GitHub
    vector_store = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=st.secrets["GROQ_API_KEY"],
        temperature=0.2
    )
    return retriever, llm

retriever, llm = init_rag()

# --- 2. UI ---
st.title("💬 How can I help you today?;")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Γράψτε την ερώτησή σας εδώ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        template = """Απάντησε στην ερώτηση χρησιμοποιώντας ΜΟΝΟ το παρακάτω context:
        {context}
        Ερώτηση: {question}
        Απάντηση:"""
        
        prompt_template = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
             "question": RunnablePassthrough()}
            | prompt_template | llm | StrOutputParser()
        )
        
        response = chain.invoke(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
