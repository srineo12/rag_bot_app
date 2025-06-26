import os
from typing import List
import pandas as pd
import streamlit as st

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_community.document_loaders import DataFrameLoader

from utils.logger import get_logger

logger = get_logger("RAGPipeline")

# --- Constants ---
PERSIST_DIR = "chroma_db_openai"
EMBED_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"
RERANK_MODEL = "rerank-english-v3.0"
TOP_K = 10
RETURN_K = 2
SCORE_THRESHOLD = 0.1
EXCEL_FILE = "data/Issue Log.xlsx"

def load_rag_pipeline():
    logger.info("🔁 Loading RAG pipeline...")

    if not st.secrets.get("OPENAI_API_KEY") or not st.secrets.get("COHERE_API_KEY"):
        logger.error("API keys missing in Streamlit secrets.")
        return None

    if not os.path.exists(EXCEL_FILE):
        logger.error(f"Excel file not found at {EXCEL_FILE}")
        return None

    # --- Load Excel as documents ---
    df = pd.read_excel(EXCEL_FILE)
    if df.empty:
        logger.error("Excel file is empty.")
        return None

    logger.debug("📄 Ingesting Excel issue log...")
    loader = DataFrameLoader(
        df,
        page_content_column="Issue Description",
        metadata_columns=["Number", "Caller", "Created", "Resolved", "Resolved by"]
    )
    docs = loader.load()

    # --- Embed and index documents ---
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, streaming=True)

    # --- Metadata-aware retriever ---
    metadata_fields = [
        AttributeInfo(name="Number", description="Incident number", type="string"),
        AttributeInfo(name="Caller", description="Issue reporter", type="string"),
        AttributeInfo(name="Created", description="Date issue was logged", type="string"),
    ]
    description = "SAP EWM issue resolution logs"

    self_query = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=description,
        metadata_field_info=metadata_fields,
        search_kwargs={"k": TOP_K}
    )

    # --- Reranker ---
    cohere_reranker = CohereRerank(
        model=RERANK_MODEL,
        top_n=RETURN_K,
        cohere_api_key=st.secrets.get("COHERE_API_KEY")
    )

    compression = ContextualCompressionRetriever(
        base_retriever=self_query,
        base_compressor=cohere_reranker
    )

    # --- Custom threshold filter ---
    class ThresholdRetriever(BaseRetriever):
        base_retriever: ContextualCompressionRetriever
        threshold: float

        def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
            logger.debug(f"Running retriever for query: {query}")
            retrieved_docs = self.base_retriever.get_relevant_documents(query)
            for doc in retrieved_docs:
                meta = doc.metadata
                doc.page_content += f"\n\n📅 Resolved On: {meta.get('Resolved', 'N/A')}\n👨‍💻 Resolved by: {meta.get('Resolved by', 'N/A')}"

            docs = self.base_retriever.invoke(query, config={"callbacks": run_manager.get_child()})
            filtered = [
                doc for doc in docs
                if doc.metadata.get("relevance_score", 0.0) >= self.threshold
            ]
            return filtered

    final = ThresholdRetriever(base_retriever=compression, threshold=SCORE_THRESHOLD)

    # --- Prompt ---
    prompt = PromptTemplate(
        template=open("prompts/qa_prompt.txt").read(),
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=final,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    logger.info("✅ RAG pipeline is ready.")
    return chain, llm
