# =============================================================================
# test_rag.py — Test the RAG Core Without the UI
# =============================================================================
# Run this BEFORE building the Streamlit UI to make sure the core works.
# If this works correctly, the Streamlit app will definitely work too.
#
# HOW TO RUN:
#   1. Make sure your .env file has the GROQ_API_KEY filled in
#   2. Put any PDF file in the project folder (e.g. your resume, any textbook)
#   3. Edit the pdf_path variable below to match your PDF filename
#   4. Run: python test_rag.py
#
# EXPECTED OUTPUT:
#   ✅ Loaded X pages from PDF
#   ✅ Split into X chunks
#   ✅ Stored X chunks in ChromaDB at './chroma_db'
#   Answer: [an actual answer about your PDF]
#   Sources used: [previews of the chunks retrieved]
# =============================================================================

from rag_core import load_and_index_pdf, build_rag_chain, ask_question

# ── EDIT THIS: path to your PDF file ─────────────────────────────────────────
# Can be any PDF: your resume, a textbook chapter, a research paper, anything
# Put the PDF in the same folder as this script and just write the filename
pdf_path = "sample.pdf"   # <-- CHANGE THIS to your actual PDF filename
# Examples:
#   pdf_path = "resume.pdf"
#   pdf_path = "chapter1.pdf"
#   pdf_path = "notes.pdf"

# ── Step 1: Index the PDF ─────────────────────────────────────────────────────
print("=" * 50)
print("STEP 1: Indexing PDF...")
print("(First time: downloads embedding model ~90MB, takes 2-3 min)")
print("(After first time: much faster)")
print("=" * 50)
vectorstore = load_and_index_pdf(pdf_path)

# ── Step 2: Build the RAG Chain ───────────────────────────────────────────────
print("\nSTEP 2: Building RAG chain (connecting to Groq API)...")
chain = build_rag_chain(vectorstore)
print("✅ Chain ready!")

# ── Step 3: Ask a Question ────────────────────────────────────────────────────
test_question = "What is this document about?"
print(f"\nSTEP 3: Asking question: '{test_question}'")
print("(Groq API call: usually takes 1-3 seconds)")

result = ask_question(chain, test_question)

# ── Step 4: Display Results ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print("ANSWER:")
print(result["answer"])

print("\nSOURCES USED (first 150 chars of each chunk):")
for i, source in enumerate(result["sources"]):
    print(f"\n  Chunk {i+1}:")
    print(f"  {source[:150]}...")

print("\n" + "=" * 50)
print("✅ Core RAG pipeline is working! Now run: streamlit run app.py")
