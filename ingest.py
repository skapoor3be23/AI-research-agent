from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Step 1: Load the PDF — one Document object per page
loader = PyPDFLoader("test.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages")

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")

# Step 3: Embed and store in FAISS
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = FAISS.from_documents(chunks, embeddings)
print("FAISS vector store built successfully")

# Step 4: Save to disk
vectorstore.save_local("faiss_index")
print("Saved to faiss_index/")
# chunk_size=1000: long enough to capture a complete thought/question,
# short enough for precise retrieval. Trade-off observed today: dense exam
# papers with multi-part questions can still split related sub-parts
# across separate chunks, since 1000 chars doesn't always capture an
# entire multi-part question block.
#
# chunk_overlap=200: helps prevent a sentence from being cut mid-way
# between chunks, but doesn't fully solve cross-chunk relationships
# (e.g. a table in one chunk, its follow-up question in the next).
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)