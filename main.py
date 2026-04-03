import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="RAG Chatbot", layout="centered")

st.title("AI PDF Chatbot (Groq Speed)")
st.markdown("Ανέβασε ένα PDF και ρώτα ό,τι θέλεις!")

# --- 1. Load Models (Cached για να μην ξαναφορτώνουν και τρώνε RAM) ---
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )
    return embeddings, llm

embeddings, llm = load_models()

with st.sidebar:
    uploaded_file = st.file_uploader("Ανέβασε το αρχείο σου", type="pdf")
    process_button = st.button("Επεξεργασία PDF")

if uploaded_file and process_button:
    with st.spinner("Διαβάζω το PDF..."):
        # Αποθήκευση προσωρινά
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        
        # Δημιουργία Vector Store στη μνήμη (για ταχύτητα)
        vector_store = Chroma.from_documents(chunks, embeddings)
        st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        st.success("Το PDF είναι έτοιμο!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_text := st.chat_input("Τι θέλεις να μάθεις;"):
    if "retriever" not in st.session_state:
        st.error("Πρέπει πρώτα να ανέβασες ένα PDF!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        with st.chat_message("assistant"):
            template = """Answer based ONLY on context: {context}\nQuestion: {question}"""
            prompt = ChatPromptTemplate.from_template(template)
            
            chain = (
                {"context": st.session_state.retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
                 "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )
            
            response = chain.invoke(prompt_text)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
