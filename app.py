# =============================================================================
# app.py — Multi-PDF Research Assistant UI (Multi-PDF)
# =============================================================================
import streamlit as st
import os
import tempfile

from rag_core import (
    load_and_index_pdfs,
    load_existing_vectorstore,
    build_rag_chain,
    ask_question,
    FAISS_DIR
)

st.set_page_config(page_title="Multi-PDF Research Assistant", page_icon="🔍", layout="centered")

st.title("🔍 Multi-PDF Research Assistant")
st.caption("Multi-PDF · FAISS · Groq LLaMA3.3 70B · LangChain")

with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🚀 Index PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            with st.spinner(f"Indexing {len(uploaded_files)} PDF(s)..."):
                tmp_paths = []
                for uf in uploaded_files:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(uf.read())
                    tmp.close()
                    tmp_paths.append((tmp.name, uf.name))

                # Rename temp files to original names so metadata is clean
                renamed_paths = []
                for tmp_path, orig_name in tmp_paths:
                    new_path = os.path.join(tempfile.gettempdir(), orig_name)
                    os.rename(tmp_path, new_path)
                    renamed_paths.append(new_path)

                vectorstore = load_and_index_pdfs(renamed_paths)
                st.session_state["chain"] = build_rag_chain(vectorstore)
                st.session_state["pdf_names"] = [uf.name for uf in uploaded_files]

                for p in renamed_paths:
                    try:
                        os.unlink(p)
                    except:
                        pass

            st.success(f"✅ Indexed {len(uploaded_files)} PDF(s)!")
            for name in st.session_state["pdf_names"]:
                st.write(f"  📄 {name}")

    st.divider()
    if st.button("⚡ Load Previous Index", use_container_width=True):
        if os.path.exists(FAISS_DIR):
            with st.spinner("Loading..."):
                vectorstore = load_existing_vectorstore()
                st.session_state["chain"] = build_rag_chain(vectorstore)
            st.success("✅ Loaded existing index!")
        else:
            st.error("No saved index found.")

    if "pdf_names" in st.session_state:
        st.divider()
        st.markdown("**📚 Indexed Documents:**")
        for name in st.session_state["pdf_names"]:
            st.write(f"• {name}")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    if "chain" not in st.session_state:
        st.error("Please upload and index PDFs first.")
    else:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask_question(st.session_state["chain"], prompt)
            st.markdown(result["answer"])
            with st.expander("📎 Source chunks retrieved (click to see)"):
                for i, chunk in enumerate(result["sources"], 1):
                    st.markdown(f"**Chunk {i}:** {chunk}")

        st.session_state["messages"].append({"role": "assistant", "content": result["answer"]})

if "chain" not in st.session_state:
    st.info("👈 Upload PDFs in the sidebar and click **Index PDFs** to begin.")
    st.markdown("""
    **💡 Try asking:**
    - *Am I eligible for this job?*
    - *What skills from my resume match the job requirements?*
    - *What skills am I missing?*
    """)
