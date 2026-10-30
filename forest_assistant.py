"""Evidence-grounded assistant page using existing EcoMRV session data."""
import importlib
import streamlit as st
import report_explanation
importlib.reload(report_explanation)
from ecosystem_pipeline import build_evidence_object, parse_historical_scores
from rag_retrieval import retrieve
answer_forest_question = report_explanation.answer_forest_question

st.title("AI Forest Assistant")
st.caption("Project answers use saved session evidence; general current questions can use Groq live web search.")
state = st.session_state
if st.button("New chat", key="forest_assistant_new_chat"):
    state["forest_assistant_messages"] = []
    st.rerun()
records = list(state.get("wildlife_evidence", {}).values())
project = str(state.get("report_project", "") or "")
zone = str(state.get("report_zone", "") or "")
try:
    history = parse_historical_scores(str(state.get("report_historical_scores", "") or ""))
except (ValueError, TypeError):
    history = []
evidence = build_evidence_object(records, project, zone, history=history)
satellite = state.get("satellite_ndvi_result")
evidence["satellite_ndvi_observations"] = satellite if satellite else "Not available in current project data."
evidence["project_metadata"] = {"project": project or "Not available in current project data.", "zone_or_location": zone or "Not available in current project data."}
documents = state.get("mrv_supporting_documents", [])
assistant_version = "public-facts-live-search-v3"
if state.get("forest_assistant_version") != assistant_version:
    state["forest_assistant_messages"] = []
    state["forest_assistant_version"] = assistant_version
messages = state.setdefault("forest_assistant_messages", [])
for item in messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("sources"): st.caption("Sources: " + ", ".join(item["sources"]))
question = st.chat_input("Ask about this forest or its monitoring evidence")
if question:
    retrieved = retrieve(documents, question, limit=4)
    context = {"ecomrv_evidence": evidence, "saved_camera_audio_satellite_records": records}
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"): st.markdown(question)
    with st.chat_message("assistant"):
        result = answer_forest_question(question, context, retrieved, messages[:-1])
        answer = result["answer"]
        if result["error"]: st.error(result["error"])
        elif result["model"]: st.caption("Groq model: " + result["model"])
        if not answer: answer = "I could not generate a response. See the configuration or Groq error above."
        st.markdown(answer)
        if not result["answer"]:
            with st.expander("Available project evidence (local, no AI answer generated)"):
                st.json(context)
        if retrieved or result.get("web_sources"):
            with st.expander("Evidence / Sources"):
                for source in retrieved:
                    st.markdown("**" + str(source.get("source", "Unknown source")) + "**")
                    st.caption(source.get("excerpt", ""))
                for source in result.get("web_sources", []):
                    st.markdown("[" + str(source["title"]) + "](" + source["url"] + ")")
                    if source.get("snippet"): st.caption(source["snippet"])
                if result.get("web_sources"): st.caption("Web sources retrieved by Groq browser search.")
        else: st.caption("Sources: current project session evidence. No supporting document matched this question.")
    messages.append({"role": "assistant", "content": answer, "sources": [d.get("source", "Unknown source") for d in retrieved] + [s["url"] for s in result.get("web_sources", [])]})
    if len(messages) > 20: del messages[:-20]










