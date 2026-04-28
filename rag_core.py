# =============================================================================
# rag_core.py — Research Assistant (Multi-PDF RAG)
# =============================================================================
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

FAISS_DIR = "./faiss_index"


def load_and_index_pdfs(pdf_paths: list):
    """
    Accepts a LIST of PDF file paths.
    Loads all PDFs, tags each chunk with source filename,
    splits into chunks, stores in one FAISS index.
    """
    all_documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        # Tag each doc with a clean source name
        for doc in docs:
            doc.metadata["source_name"] = os.path.basename(pdf_path)
        all_documents.extend(docs)
        print(f"  loaded: {os.path.basename(pdf_path)} ({len(docs)} pages)")

    print(f"Total pages: {len(all_documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(all_documents)
    print(f"Split into {len(chunks)} chunks")

    vectorstore = FAISS.from_documents(documents=chunks, embedding=EMBEDDING_MODEL)
    vectorstore.save_local(FAISS_DIR)
    print(f"Saved FAISS index at {FAISS_DIR}")
    return vectorstore


def load_and_index_pdf(pdf_path: str):
    """Single PDF wrapper."""
    return load_and_index_pdfs([pdf_path])


def load_existing_vectorstore():
    """Loads previously saved FAISS index from disk."""
    return FAISS.load_local(
        FAISS_DIR,
        EMBEDDING_MODEL,
        allow_dangerous_deserialization=True
    )


def build_rag_chain(vectorstore):
    """Builds RetrievalQA chain with Groq LLaMA3.3 70B."""
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    prompt_template = """You are a helpful Research Assistant with access to multiple documents.
The context below contains chunks from different documents (e.g. a job description and a resume).
Each chunk may be labelled with its source.

Your task:
- Read ALL chunks carefully before answering.
- If the question involves comparing documents (e.g. eligibility, match, fit), explicitly cross-reference information from BOTH documents.
- List matching skills, qualifications, or facts you find.
- Be specific. Quote or paraphrase directly from the context.
- Only say "Not found" if the information is genuinely absent from ALL chunks.

Context:
{context}

Question: {question}

Answer (cross-reference both documents if relevant, be detailed):"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # k=8 so we pull chunks from BOTH documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain


def ask_question(chain, question: str) -> dict:
    """Runs the chain and returns answer + source chunks."""
    result = chain({"query": question})
    return {
        "answer": result["result"],
        "sources": [
            f"[{doc.metadata.get('source_name', 'doc')}] {doc.page_content[:200]}"
            for doc in result["source_documents"]
        ]
    }
