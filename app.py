import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import assistant
from ingest import build_vectorstore

load_dotenv()

st.set_page_config(page_title="AI Research Assistant", page_icon="🔎")

st.title("🔎 AI Research Assistant")
st.caption("Upload one or more PDFs, then ask questions from your documents, the web, or general knowledge - the agent decides where to look.")

if "history" not in st.session_state:
    st.session_state.history = []

if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = assistant.vectorstore is not None

with st.sidebar:
    st.subheader("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if not uploaded_files:
            st.warning("Please select at least one PDF first.")
        else:
            with st.spinner("Reading and indexing documents..."):
                temp_dir = tempfile.mkdtemp()
                temp_paths = []
                for f in uploaded_files:
                    path = os.path.join(temp_dir, f.name)
                    with open(path, "wb") as out:
                        out.write(f.getbuffer())
                    temp_paths.append(path)

                new_vectorstore, chunk_count = build_vectorstore(temp_paths)
                assistant.vectorstore = new_vectorstore
                st.session_state.docs_loaded = True

            st.success(f"Indexed {chunk_count} chunks from {len(uploaded_files)} document(s).")

    if st.session_state.docs_loaded:
        st.info("Documents are loaded and ready to query.")
    else:
        st.warning("No documents loaded yet. Upload a PDF above to enable document Q&A.")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])
        st.caption("Source: " + turn["source"] + " | Citations: " + str(turn["citations"]))

question = st.chat_input("Ask anything...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Thinking..."):
        ans, source_used, citations = assistant.answer(question, st.session_state.history)

    with st.chat_message("assistant"):
        st.write(ans)
        st.caption("Source: " + source_used + " | Citations: " + str(citations))

    st.session_state.history.append({
        "q": question,
        "a": ans,
        "source": source_used,
        "citations": citations
    })