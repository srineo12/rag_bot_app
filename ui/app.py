import os
import sys
import re
import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from retrievers.rag_pipeline import load_rag_pipeline

# --- Load environment ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# --- Streamlit setup ---
st.set_page_config(page_title="AI Issue Resolution Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AI Issue Resolution Assistant")
st.info("Chatbot uses SAP EWM logs to provide resolution summaries.")

# --- Load RAG pipeline ---
st.write("✅ Keys Found:", st.secrets.get("OPENAI_API_KEY") is not None, st.secrets.get("COHERE_API_KEY") is not None)
st.write("✅ File Exists:", os.path.exists("data/Issue Log.xlsx"))

result = load_rag_pipeline()
if result is None:
    st.error("Chatbot not available. API keys or database might be missing.")
    st.stop()

qa_chain, llm = result


# --- Session state for history ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

# --- Display past messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# --- User input ---
if prompt := st.chat_input("Ask about an issue..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if not qa_chain:
                response = "🚫 Chatbot is not available due to configuration error."
            else:
                result = qa_chain.invoke({"query": prompt})
                docs = result.get("source_documents", [])

                if not docs:
                    response = "⚠️ No relevant matches found. Try rephrasing your question."
                else:
                    response = ""
                    for i, doc in enumerate(docs, start=1):
                        meta = doc.metadata
                        ticket_no = meta.get("Number", "N/A")
                        resolved = meta.get("Resolved", "N/A")
                        resolved_by = meta.get("Resolved_by", "N/A")
                        score = round(meta.get("relevance_score", 0.0), 2)

                        trimmed_content = doc.page_content.strip()

                        # Default fallback
                        first_paragraph = trimmed_content
                        second_paragraph = ""

                        try:
                            polish_prompt = f"""Please polish the following SAP incident summary by fixing grammar, clarity, and structure. Split it into two sections: 
1. Issue Description 
2. Resolution 
Make both sections clear and easy to understand. Do not repeat or reprint the incident number:
{trimmed_content}"""
                            polished = llm.invoke(polish_prompt).content.strip()

                            # Attempt to split the polished result
                            if "Resolution:" in polished:
                                parts = polished.split("Resolution:", 1)
                              

                                first_paragraph = re.sub(r"\*{1,2}", "", parts[0]).strip()
                                first_paragraph = re.sub(r"(?i)^issue description[:\s\-]*", "", first_paragraph).strip()

                                second_paragraph = re.sub(r"\*{1,2}", "", parts[1]).strip()
                                second_paragraph = re.sub(r"(?i)^resolution[:\s\-]*", "", second_paragraph).strip()

                            else:
                                first_paragraph = polished.strip()

                        except Exception as e:
                            st.warning(f"⚠️ LLM polishing failed: {e}")

                        # Final HTML formatting
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
        st.session_state.messages.append({"role": "assistant", "content": response})

