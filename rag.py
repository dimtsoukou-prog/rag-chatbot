import streamlit as st
import os
from rag import extract_text_from_pdf, split_text, create_vector_store, load_rag_chain
from dotenv import load_dotenv

load_dotenv()

# Ρύθμιση σελίδας για να φαίνεται ωραίο στο WordPress iframe
st.set_page_config(page_title="PDF Chatbot", layout="centered")

st.title("🤖 AI Assistant")
st.info("Ανέβασε ένα PDF για να ξεκινήσουμε τη συζήτηση.")

# Χρησιμοποιούμε cache για να μην φορτώνει το μοντέλο σε κάθε κλικ
@st.cache_resource
def get_chain():
    # Αν δεν υπάρχει ήδη το index, επιστρέφουμε None
    if not os.path.exists("faiss_index"):
        return None
    chain, retriever = load_rag_chain()
    return chain

# --- Sidebar για το Upload ---
with st.sidebar:
    st.header("Ρυθμίσεις")
    uploaded_file = st.file_uploader("Επιλογή PDF", type="pdf")
    
    if st.button("Ανάλυση Αρχείου"):
        if uploaded_file:
            with st.spinner("Γίνεται επεξεργασία..."):
                # Αποθήκευση και επεξεργασία
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                text = extract_text_from_pdf("temp.pdf")
                chunks = split_text(text)
                create_vector_store(chunks)
                st.success("Έτοιμο! Τώρα μπορείς να ρωτήσεις.")
                st.rerun() # Refresh για να δει το νέο index

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Εμφάνιση ιστορικού
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Ερώτηση χρήστη
if prompt := st.chat_input("Πώς μπορώ να βοηθήσω;"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Φόρτωση του chain
    chain = get_chain()
    
    if chain:
        with st.chat_message("assistant"):
            with st.spinner("Σκέφτομαι..."):
                response = chain.invoke(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Παρακαλώ ανέβασε πρώτα ένα αρχείο PDF από το πλάι.")
