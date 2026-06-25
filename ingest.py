import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def build_vectorstore(file_paths):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = []

    for path in file_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        chunks = splitter.split_documents(pages)

        filename = os.path.basename(path)
        for chunk in chunks:
            chunk.metadata["filename"] = filename

        all_chunks.extend(chunks)

    if len(all_chunks) == 0:
        raise ValueError(
            "No text could be extracted from the uploaded file(s). "
            "This usually means the PDF is a scanned image with no "
            "selectable text. Try a different PDF that has real text "
            "(not just scanned pages)."
        )

    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    return vectorstore, len(all_chunks)


if __name__ == "__main__":
    vectorstore, chunk_count = build_vectorstore(["test.pdf"])
    print(f"Created {chunk_count} chunks")
    vectorstore.save_local("faiss_index")
    print("Saved to faiss_index/")