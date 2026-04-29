<<<<<<< Updated upstream
# 🔍 Multi-PDF Research Assistant
=======
A multi-PDF question-answering chatbot built with LangChain, FAISS, Groq LLaMA3.3 70B, and Streamlit.

Upload one or more PDFs and ask questions — the assistant cross-references all documents to give detailed, sourced answers. Conversation memory is maintained across turns, so follow-up questions work naturally.

---

## 🚀 Features

- **Multi-PDF support** — upload and query multiple PDFs simultaneously
- **Semantic search** — FAISS vector index with HuggingFace embeddings
- **Conversation memory** — follow-up questions retain context from earlier turns
- **Fast LLM responses** — Groq-hosted LLaMA3.3 70B (~276 tokens/sec)
- **Source transparency** — see exactly which chunks were used to answer
- **Cross-document reasoning** — compare a resume vs job description, or multiple research papers

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Groq LLaMA3.3 70B Versatile |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (via langchain-huggingface) |
| Vector Store | FAISS (local, CPU) |
| Orchestration | LangChain ConversationalRetrievalChain |
| Memory | LangChain ConversationBufferMemory |
| UI | Streamlit |
| PDF Loader | PyPDFLoader |

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/shrutimaurya11/Multi-PDF-Research-Assistant.git
cd Multi-PDF-Research-Assistant

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
# Create a .env file and add:
GROQ_API_KEY=your_key_here

# 5. (Optional) Run smoke test first
python test_rag.py

# 6. Run the app
streamlit run app.py
```

Get a free Groq API key at: https://console.groq.com

---

## 💡 Example Use Cases

- Upload a **job description + resume** → ask *"Am I eligible for this role?"*
- Upload **multiple research papers** → ask *"What are the common findings?"*
- Upload any **PDF document** → ask follow-up questions across turns

---

## 📁 Project Structure

```
Multi-PDF-Research-Assistant/
├── app.py              # Streamlit UI
├── rag_core.py         # RAG pipeline (load, embed, retrieve, answer, memory)
├── test_rag.py         # Smoke test — run before the UI to verify setup
├── requirements.txt    # Pinned dependencies
├── .env                # API key (not committed — add to .gitignore)
└── .gitignore
```

---

## 🧠 How It Works

1. PDFs are loaded page by page using PyPDFLoader
2. Text is split into 700-character chunks with 100-char overlap
3. Each chunk is embedded into a 384-dim vector using MiniLM-L6-v2
4. Vectors are stored in a local FAISS index (persisted to `./faiss_index`)
5. On each question, the top 8 most similar chunks are retrieved
6. Chunks + full conversation history are sent to Groq LLaMA3.3 70B
7. The LLM cross-references all chunks and returns a detailed, sourced answer

---

## ⚠️ Notes

- Re-indexing PDFs will **overwrite** the existing FAISS index and clear chat history
- Max supported PDF size: **20 MB per file**
- The embedding model (~90MB) is downloaded on first run and cached automatically
