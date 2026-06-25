import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from router import route_question
from search import web_search

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

RELEVANCE_THRESHOLD = 0.7

# Starts as None. If a saved index exists on disk (built via
# "python ingest.py"), it loads automatically so the CLI and API
# keep working. The Streamlit app can replace this at runtime after
# a user uploads PDFs - that's what enables multi-document support.
vectorstore = None
if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


def rewrite_question(question, history):
    if not history:
        return question

    recent = history[-2:]
    lines = []
    for h in recent:
        lines.append("Q: " + h["q"])
        lines.append("A: " + h["a"])
    history_text = "\n".join(lines)

    prompt = (
        "Conversation history:\n" + history_text + "\n\n"
        "Follow-up question: " + question + "\n\n"
        "Rewrite the follow-up question as a standalone question by replacing "
        "pronouns (it, its, that, this, etc.) with the actual subject from the "
        "history. If it is already standalone, return it unchanged. "
        "Return ONLY the rewritten question, nothing else."
    )

    response = llm.invoke(prompt)
    return response.content.strip()


def get_history_context(history):
    if not history:
        return ""

    if len(history) <= 3:
        lines = []
        for h in history:
            lines.append("Q: " + h["q"])
            lines.append("A: " + h["a"])
        recent_text = "\n".join(lines)
        return "Recent conversation:\n" + recent_text + "\n\n"

    recent = history[-3:]
    older = history[:-3]

    older_lines = []
    for h in older:
        older_lines.append("Q: " + h["q"])
        older_lines.append("A: " + h["a"])
    older_text = "\n".join(older_lines)

    summary = llm.invoke("Summarise this conversation in 2 short sentences:\n" + older_text)

    recent_lines = []
    for h in recent:
        recent_lines.append("Q: " + h["q"])
        recent_lines.append("A: " + h["a"])
    recent_text = "\n".join(recent_lines)

    return "Earlier: " + summary.content + "\n\nRecent conversation:\n" + recent_text + "\n\n"


def document_answer(question, history_context):
    if vectorstore is None:
        return (
            "No documents have been loaded yet. Upload a PDF in the "
            "sidebar first, or run ingest.py to build a local index."
        ), []

    results = vectorstore.similarity_search_with_score(question, k=3)
    best_score = results[0][1]

    if best_score > RELEVANCE_THRESHOLD:
        return (
            "I couldn't find relevant information in your documents for "
            "this question. Try rephrasing it, or use a web search instead."
        ), []

    docs = [doc for doc, score in results]
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = (
        history_context
        + "Answer the question using only the context below.\n"
        + 'If the answer isn\'t in the context, say "I couldn\'t find this in the document."\n\n'
        + "Context:\n" + context + "\n\n"
        + "Question: " + question
    )

    response = llm.invoke(prompt)
    sources = []
    for doc in docs:
        filename = doc.metadata.get("filename") or doc.metadata.get("source", "document")
        filename = os.path.basename(filename)
        page = doc.metadata.get("page", "?")
        sources.append(filename + " - Page " + str(page))

    return response.content, sources


def web_answer(question, history_context):
    search_results = web_search(question)

    prompt = (
        history_context
        + "Answer the question using the web search results below.\n"
        + "If the results don't contain a clear answer, say so honestly.\n\n"
        + "Web Search Results:\n" + search_results + "\n\n"
        + "Question: " + question
    )

    response = llm.invoke(prompt)
    return response.content, ["Web search"]


def general_answer(question, history_context):
    prompt = history_context + "Question: " + question
    response = llm.invoke(prompt)
    return response.content, ["General knowledge"]


def answer(question, history):
    standalone_question = rewrite_question(question, history)
    history_context = get_history_context(history)
    category = route_question(standalone_question)

    if category == "document":
        ans, sources = document_answer(standalone_question, history_context)
    elif category == "web":
        ans, sources = web_answer(standalone_question, history_context)
    else:
        ans, sources = general_answer(standalone_question, history_context)

    return ans, category, sources


if __name__ == "__main__":
    chat_history = []
    print("AI Research Assistant - type 'exit' to quit\n")

    while True:
        question = input("You: ")
        if question.lower() in ("exit", "quit"):
            break

        ans, source_used, sources = answer(question, chat_history)

        print("\n[Source: " + source_used + "]")
        print("Assistant: " + ans)
        print("Citations: " + str(sources) + "\n")

        chat_history.append({"q": question, "a": ans})