from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from assistant import answer

load_dotenv()

app = FastAPI(title="AI Research Assistant API")

sessions = {}


class Query(BaseModel):
    question: str
    session_id: str = "default"


class Response(BaseModel):
    answer: str
    source_used: str
    citations: list[str]


@app.post("/ask", response_model=Response)
def ask(query: Query):
    history = sessions.get(query.session_id, [])

    ans, source_used, citations = answer(query.question, history)

    history.append({"q": query.question, "a": ans})
    sessions[query.session_id] = history

    return Response(answer=ans, source_used=source_used, citations=citations)


@app.get("/")
def root():
    return {"message": "AI Research Assistant API is running. POST to /ask to query it."}