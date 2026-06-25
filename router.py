from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def route_question(question):
    prompt = f"""Classify this question into exactly one category.
Return only the single word, nothing else - no punctuation, no explanation.

Categories:
- document: question about content from an uploaded file, notes, or specific stored information
- web: question needing recent news, current events, or live/real-time data
- general: factual question answerable from general knowledge, not needing live data or specific documents

Question: {question}

Category:"""

    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    return category

# Test it with several questions - only runs when this file is run directly,
# not when imported by another file like assistant.py
if __name__ == "__main__":
    test_questions = [
        "What is process P1's resource allocation?",
        "What is the latest news about AI regulations?",
        "What is the capital of France?",
        "Who won the cricket match yesterday?",
        "Explain what a deadlock is in operating systems",
    ]

    for q in test_questions:
        category = route_question(q)
        print(f"Q: {q}")
        print(f"   -> Routed as: {category}\n")
