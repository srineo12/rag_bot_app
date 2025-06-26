import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

def sanitize_metadata(metadata: dict) -> dict:
    """Ensure all metadata values are str, int, float, bool, or None."""
    clean = {}
    for k, v in metadata.items():
        if pd.isna(v):
            continue
        if isinstance(v, (pd.Timestamp, pd.Timedelta)):
            clean[k] = str(v)
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean

def ingest():
    if not os.getenv("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY missing in .env")
        sys.exit(1)
    if not os.path.exists(EXCEL_FILE):
        log.error(f"Excel file not found: {EXCEL_FILE}")
        sys.exit(1)

    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET)
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
            if col not in CONTENT_COLS:
                metadata[col.replace(" ", "_")] = row.get(col)

        sanitized = sanitize_metadata(metadata)
        docs.append(Document(page_content=page_content, metadata=sanitized))

    embeddings = OpenAIEmbeddings(model=EMB_MODEL)
    Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)

    log.info("✅ Ingestion complete – %d documents stored.", len(docs))

if __name__ == "__main__":
    ingest()
