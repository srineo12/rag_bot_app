import os, sys
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from utils.logger import get_logger

# --- Load environment ---
load_dotenv()
log = get_logger(__name__)

# --- Constants ---
EXCEL_FILE   = "data/Issue Log.xlsx"
SHEET        = 0
CONTENT_COLS = ["Description", "Resolution", "Number"]
PERSIST_DIR  = "chroma_db_openai"
EMB_MODEL    = "text-embedding-3-large"

def ingest():
    if not os.getenv("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY missing in .env")
        sys.exit(1)
    if not os.path.exists(EXCEL_FILE):
        log.error(f"Excel file not found: {EXCEL_FILE}")
        sys.exit(1)

    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET)

    # Format any datetime columns
    for col in df.select_dtypes(include=["datetime64"]).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%d")

    log.info("Loaded %d rows from Excel", len(df))

    docs = []
    for idx, row in df.iterrows():
        incident_number = str(row.get("Number")).strip() if pd.notna(row.get("Number")) else "N/A"
        description = row.get("Description", "")
        resolution = row.get("Resolution", "")

        page_content = f"""Incident Number: {incident_number}

Issue Description: {description}

Resolution: {resolution}
"""

        metadata = {
            "row_id": idx,
            "Number": incident_number,
            "Resolved": row.get("Resolved", "N/A"),
            "Resolved_by": row.get("Resolved_by", "N/A"),
        }

        # Add any other metadata columns dynamically
        for col in df.columns:
            if col not in CONTENT_COLS and pd.notna(row[col]):
                metadata[col.replace(" ", "_")] = row[col]

        docs.append(Document(page_content=page_content, metadata=metadata))

    # Each doc is a single row — no chunking
    embeddings = OpenAIEmbeddings(model=EMB_MODEL)
    Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)

    log.info("Ingestion complete – %d documents stored.", len(docs))

if __name__ == "__main__":
    ingest()
