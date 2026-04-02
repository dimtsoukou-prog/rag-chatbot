import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

app = FastAPI(title="Groq RAG Chatbot 2026")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTOR_DB_DIR = "vector_db"

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

retriever = None

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global retriever
    try:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTOR_DB_DIR
        )

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        os.remove(file_path)
        return {"message": f"File {file.filename} processed! {len(chunks)} chunks created."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    global retriever
    if retriever is None:
        raise HTTPException(status_code=400, detail="Please upload a PDF first!")

    try:
        prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context:
        {context}

        Question: {question}

        Helpful Answer:""")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke(request.question)
        docs = retriever.invoke(request.question)

        return {
            "question": request.question,
            "answer": answer,
            "sources": [doc.page_content[:200] for doc in docs]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
def root():
    return {"status": "RAG Chatbot is running! 🚀"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
