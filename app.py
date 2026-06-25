import streamlit as st
from dotenv import load_dotenv
from assistant import answer

load_dotenv()

st.set_page_config(page_title="AI Research Assistant", page_icon="🔎")

st.title("🔎 AI Research Assistant")
st.caption("Ask questions from your documents, the web, or general knowledge — the agent decides where to look.")

# Session state stores the conversation history for this browser session
if "history" not in st.session_state:
    st.session_state.history = []

# Display past conversation
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])
        st.caption(f"Source: {turn['source']} | Citations: {turn['citations']}")

# Input box at the bottom
question = st.chat_input("Ask anything...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.spinner("Thinking..."):
        ans, source_used, citations = answer(question, st.session_state.history)

    with st.chat_message("assistant"):
        st.write(ans)
        st.caption(f"Source: {source_used} | Citations: {citations}")

    st.session_state.history.append({
        "q": question,
        "a": ans,
        "source": source_used,
        "citations": citations
    })