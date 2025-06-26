import os
import sys
import re
import streamlit as st

# ✅ Page config must be first
st.set_page_config(page_title="AI Issue Resolution Assistant", page_icon="🤖", layout="wide")

# ✅ Show key and file checks
st.write("🔐 OPENAI key found:", "OPENAI_API_KEY" in st.secrets)
st.write("🔐 COHERE key found:", "COHERE_API_KEY" in st.secrets)
st.write("✅ File Exists:", os.path.exists("data/Issue Log.xlsx"))

# ✅ For debugging Streamlit Cloud
st.write("📂 Current working dir:", os.getcwd())
st.write("📄 Files in root:", os.listdir("."))

# --- Add project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Load environment
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# --- TEMP ingestion button
if st.button("🚀 Run ingestion (temporary button for Streamlit Cloud)"):
    try:
        from data_ingestion.ingest_excel import ingest
        ingest()
        st.success("✅ Ingestion complete. Vectorstore created.")
    except Exception as e:
        st.error(f"❌ Ingestion failed: {e}")
        st.stop()

# --- Load RAG pipeline
from retrievers.rag_pipeline import load_rag_pipeline

result = load_rag_pipeline()
if result is None:
    st.error("Chatbot not available. API keys or database might be missing.")
    st.stop()

qa_chain, llm = result

# --- Chat UI
st.title("🤖 AI Issue Resolution Assistant")
st.info("Chatbot uses SAP EWM logs to provide resolution summaries.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Ask about an issue..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke({"query": prompt})
                docs = result.get("source_documents", [])

                if not docs:
                    response = "⚠️ No relevant matches found."
                else:
                    response = ""
                    for i, doc in enumerate(docs, start=1):
                        meta = doc.metadata
                        ticket_no = meta.get("Number", "N/A")
                        resolved = meta.get("Resolved", "N/A")
                        resolved_by = meta.get("Resolved_by", "N/A")
                        score = round(meta.get("relevance_score", 0.0), 2)

                        trimmed_content = doc.page_content.strip()

                        # Polishing with LLM
                        polish_prompt = f"""Please polish the following SAP incident summary by fixing grammar, clarity, and structure. Split it into two sections: 
1. Issue Description 
2. Resolution 
Make both sections clear and easy to understand. Do not repeat or reprint the incident number:
{trimmed_content}"""

                        try:
                            polished = llm.invoke(polish_prompt).content.strip()
                            if "Resolution:" in polished:
                                parts = polished.split("Resolution:", 1)
                                first_paragraph = re.sub(r"(?i)^issue description[:\s\-]*", "", parts[0]).strip()
                                second_paragraph = re.sub(r"(?i)^resolution[:\s\-]*", "", parts[1]).strip()
                            else:
                                first_paragraph = polished.strip()
                                second_paragraph = ""
                        except Exception as e:
                            first_paragraph = trimmed_content
                            second_paragraph = ""
                            st.warning(f"⚠️ LLM polishing failed: {e}")

                        response += f"""
<div style='margin-bottom: 20px; padding: 10px; border-bottom: 1px solid #ddd'>
  🆔 <strong>Incident Number:</strong> {ticket_no} |
  📅 <strong>Resolved On:</strong> {resolved} |
  👨‍💻 <strong>Resolved By:</strong> {resolved_by} |
  ⭐ <strong>Relevance Score:</strong> {score:.2f}
  <p style='font-weight: normal; font-size: 15px; margin-top: 10px'>
    <strong>Issue Description:</strong> {first_paragraph}<br><br>
    <strong>Resolution:</strong> {second_paragraph}
  </p>
</div>
"""
                st.markdown(response, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Query failed: {e}")

        st.session_state.messages.append({"role": "assistant", "content": response})
