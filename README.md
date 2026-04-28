# # 📚 Multi-PDF Research Assistant

A multi-PDF question-answering chatbot built with LangChain, FAISS, Groq LLaMA3.3 70B, and Streamlit.

Upload one or more PDFs and ask questions — the assistant cross-references all documents to give detailed, sourced answers.

---

## 🚀 Features

- **Multi-PDF support** — upload and query multiple PDFs simultaneously
- **Semantic search** — FAISS vector index with HuggingFace embeddings
- **Fast LLM responses** — Groq-hosted LLaMA3.3 70B (~276 tokens/sec)
- **Source transparency** — see exactly which chunks were used to answer
- **Cross-document reasoning** — compare a resume vs job description, or multiple research papers

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq LLaMA3.3 70B Versatile |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | FAISS (local) |
| Orchestration | LangChain RetrievalQA |
| UI | Streamlit |
| PDF Loader | PyPDFLoader |

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOURUSERNAME/rag-pdf-chatbot.git
cd rag-pdf-chatbot

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
# Create a .env file and add:
GROQ_API_KEY=your_key_here

# 5. Run the app
streamlit run app.py
```

Get a free Groq API key at: https://console.groq.com

---

## 💡 Example Use Cases

- Upload a **job description + resume** → ask *"Am I eligible for this role?"*
- Upload **multiple research papers** → ask *"What are the common findings?"*
- Upload any **PDF document** → ask questions in plain English

---

## 📁 Project Structure

```
rag-pdf-chatbot/
├── app.py              # Streamlit UI
├── rag_core.py         # RAG pipeline (load, embed, retrieve, answer)
├── test_rag.py         # Quick core test
├── requirements.txt    # Dependencies
├── .env                # API key (not committed)
└── .gitignore
```

---

## 🧠 How It Works

1. PDFs are loaded page by page using PyPDFLoader
2. Text is split into 700-character chunks with 100-char overlap
3. Each chunk is converted to a 384-dim vector using MiniLM-L6-v2
4. Vectors are stored in a local FAISS index
5. On each question, the top 8 most similar chunks are retrieved
6. Chunks + question are sent to Groq LLaMA3.3 70B
7. The LLM cross-references all chunks and returns a detailed answer
