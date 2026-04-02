## An AI-powered chatbot that answers questions from your own PDF documents using Retrieval-Augmented Generation (RAG).
## How It Works
1. Upload any PDF document
2. The AI reads and indexes it
3. Ask questions and get answers based on your document

## Tech Stack
- **Backend:** FastAPI + Python
- **AI Model:** Llama 3.3 70B via Groq API
- **Embeddings:** HuggingFace sentence-transformers
- **Vector Store:** ChromaDB
- **Deployment:** Render

##  Installation

1. Clone the repo:
```bash
git clone https://github.com/dimtsoukou-prog/rag-chatbot.git
cd rag-chatbot
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
