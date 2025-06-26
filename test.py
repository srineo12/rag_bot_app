import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

db = Chroma(
    persist_directory="chroma_db_openai",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-large")
)
print(f"✅ Documents in DB: {len(db.get()['documents'])}")