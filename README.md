# AI Research Assistant - Multi-Source RAG with Intelligent Routing

An AI-powered research assistant that intelligently decides where to look before answering - your documents, the live web, or general knowledge - instead of relying on a single static source.

## The Problem It Solves

Researchers and students constantly switch between their own notes, the web, and general knowledge while researching a topic. This assistant unifies all three into a single conversational interface, automatically routing each question to the right source.

## How It Works - Architecture

1. Query Rewriting - Follow-up questions are rewritten into standalone questions using conversation history, so retrieval doesn't fail on pronouns.
2. Intelligent Routing - A Gemini-powered classifier decides whether a question needs document retrieval, a live web search, or general knowledge.
3. RAG Pipeline - Documents are loaded, split into 1000-character chunks (200-character overlap), embedded with Gemini's embedding model, and stored in a FAISS vector database.
4. Hallucination Prevention - Before generating an answer, the system checks the relevance score of retrieved chunks. If the best match is too dissimilar, it declines to answer instead of guessing.
5. Conversation Memory - Recent exchanges are kept in full; older ones are compressed into a short summary, preventing context window overflow.
6. REST API Layer - The entire pipeline is exposed via a FastAPI /ask endpoint with session-based history.

## Features

- Multi-source intelligent routing (document / web / general)
- Grounded document retrieval with page-level citations
- Live web search integration (DuckDuckGo)
- Context-compressed conversation memory
- Query rewriting for natural follow-up questions
- Hallucination prevention via relevance thresholding
- REST API with transparent source attribution
- Streamlit chat interface

## Tech Stack

- Python - Core language for all AI/ML tooling
- LangChain - Document loaders, text splitters, vector store integration
- Gemini API - Free tier, used for embeddings and generation
- FAISS - Local, fast similarity search, no API key needed
- Streamlit - Quick, clean chat UI
- FastAPI - Production-style REST API with auto-generated docs

## Setup

Clone the repository and install dependencies:

    pip install -r requirements.txt

Create a .env file with your Gemini API key (see .env.example).

Add your own PDF as test.pdf, then run:

    python ingest.py

Run the chat interface:

    streamlit run app.py

Or run the REST API:

    uvicorn api:app --reload

Visit http://127.0.0.1:8000/docs for interactive API documentation.

## Known Limitations

- Similarity search can miss information split across non-adjacent chunks in densely structured documents.
- The free-tier web search API works best for well-known topics; limited coverage for breaking news.
- Conversation memory is session-based and does not persist after restart.
- The system answers only from information explicitly stated in the
  document. It does not perform calculations the document leaves as
  an exercise (e.g. "calculate the profit if output is 20 units" when
  only the cost function is given) - it correctly declines rather than
  guessing.

## Credits

Built as part of a self-directed AI/ML learning sprint. Architecture and individual features were researched, implemented, and tested independently.
