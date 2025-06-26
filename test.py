import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# --- Config ---
PERSIST_DIR = "chroma_db_openai"
EMB_MODEL = "text-embedding-3-large"
MAX_DOCS = 50  # Limit display for readability

# --- Initialize ---
embeddings = OpenAIEmbeddings(model=EMB_MODEL)
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# --- Fetch stored documents ---
docs = vectorstore.similarity_search("test", k=MAX_DOCS)

print(f"\n🔍 Found {len(docs)} documents in vector store:\n")

for i, doc in enumerate(docs, start=1):
    print(f"--- Document #{i} ---")
    print("📄 Content (first 200 chars):", doc.page_content[:200].strip())
    print("📌 Metadata:")
    for key, value in doc.metadata.items():
        print(f"  • {key}: {value}")
    print("-" * 50)

print("\n✅ Done. Review column names and metadata values above.\n")
