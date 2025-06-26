import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Configuration
PERSIST_DIRECTORY = "chroma_db_openai"
EMBEDDING_MODEL_NAME = "text-embedding-3-large"

# Load vectorstore
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

# Access raw documents
docs = vectorstore.get()["documents"]
metas = vectorstore.get()["metadatas"]

# Display document count
print(f"🔍 Total documents in vectorstore: {len(docs)}\n")

# Display content and metadata (first 5)
for i in range(min(5, len(docs))):
    print(f"--- Document #{i+1} ---")
    print(docs[i])
    print("Metadata:", metas[i])
    print()
