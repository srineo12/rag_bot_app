import os
from typing import List
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

def load_rag_pipeline():
    logger.info("🔁 Loading RAG pipeline...")

    # --- Debug logs for deployment ---
    openai_ok = bool(st.secrets.get("OPENAI_API_KEY"))
    cohere_ok = bool(st.secrets.get("COHERE_API_KEY"))
    db_ok = os.path.exists(PERSIST_DIR)

    st.write(f"✅ Keys: OPENAI: {openai_ok}, COHERE: {cohere_ok}")
    st.write(f"✅ Chroma DB exists: {db_ok}")
    logger.debug(f"OPENAI: {st.secrets.get('OPENAI_API_KEY')}")
    logger.debug(f"COHERE: {st.secrets.get('COHERE_API_KEY')}")
    logger.debug(f"DB Path Exists: {os.path.exists(PERSIST_DIR)}")

    if not openai_ok or not cohere_ok:
        logger.error("API keys missing in Streamlit secrets.")
        return None
    if not db_ok:
        logger.error(f"Chroma DB not found at {PERSIST_DIR}")
        return None

    # --- Load components ---
    logger.debug("🔌 Loading vectorstore, LLM, and embeddings...")
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, streaming=True)

    # --- Metadata-aware retriever ---
    logger.debug("📑 Building metadata-aware retriever...")
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

    # --- Custom Threshold Retriever ---
    class ThresholdRetriever(BaseRetriever):
        base_retriever: ContextualCompressionRetriever
        threshold: float

        def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
            logger.debug(f"Running retriever for query: {query}")

            retrieved_docs = self.base_retriever.get_relevant_documents(query)
            for doc in retrieved_docs:
                meta = doc.metadata
                resolved_info = f"\n\n📅 Resolved On: {meta.get('Resolved', 'N/A')}\n👨‍💻 Resolved by: {meta.get('Resolved_by', 'N/A')}"
                doc.page_content += resolved_info
            logger.debug(f"Enriched {len(retrieved_docs)} docs with metadata.")

            docs = self.base_retriever.invoke(query, config={"callbacks": run_manager.get_child()})
            logger.debug(f"Retrieved {len(docs)} docs before threshold filter")
            for i, doc in enumerate(docs):
                logger.debug(f"[Reranked #{i+1}] Score: {doc.metadata.get('relevance_score', 0.0):.3f}, Number: {doc.metadata.get('Number', 'N/A')}")

            filtered = [
                doc for doc in docs
                if doc.metadata.get("relevance_score", 0.0) >= self.threshold
            ]
            logger.debug(f"Returning {len(filtered)} docs after threshold {self.threshold}")
            return filtered

    final_retriever = ThresholdRetriever(base_retriever=compression, threshold=SCORE_THRESHOLD)

    # --- Prompt ---
    logger.debug("📄 Loading prompt template...")
    prompt = PromptTemplate(
        template=open("prompts/qa_prompt.txt").read(),
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=final_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    logger.info("✅ RAG pipeline is ready.")
    return chain, llm
