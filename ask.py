from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def rag_answer(question):
    docs = vectorstore.similarity_search(question, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    print("\n--- DEBUG: RETRIEVED CONTEXT ---")
    print(context)
    print("--- END DEBUG ---\n")

    prompt = f"""Answer the question using only the context below.
If the answer isn't in the context, say "I couldn't find this in the document."

Context:
{context}

Question: {question}"""

    response = llm.invoke(prompt)
    return response.content, docs

question = input("Ask a question about your document: ")
answer, sources = rag_answer(question)

print("\n--- ANSWER ---")
print(answer)

print("\n--- SOURCES ---")
for doc in sources:
    page = doc.metadata.get("page", "unknown")
    print(f"Page {page}")