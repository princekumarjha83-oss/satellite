"""Combined report from user-provided wildlife media, separate from demo reports."""
import json
from datetime import date

import streamlit as st

from wildlife_reporting import KINDS, build_report, import_session_evidence, markdown_report
from rag_retrieval import ingest_document, retrieve
from ecosystem_pipeline import parse_historical_scores
from ecomrv_config import FUSION_WEIGHTS
from ui_state import saved_control
from report_explanation import available_groq_models

st.header("Wildlife Analysis Report")
st.caption("Combined evidence from your audio, video, camera analyses and uploaded satellite images in this session.")
with st.expander("EcoMRV workflow", expanded=True):
    st.code("FOREST DATA → CAMERA | AUDIO | SATELLITE → FEATURES → NORMALIZATION → FUSION → "
            "AGREEMENT | ANOMALY | ECOSYSTEM → PROPOSED MEHS → EVIDENCE + RETRIEVAL → "
            "GROUNDED EXPLANATION → MRV REPORT", language=None)
    st.caption("Existing models are retained: OWL-ViT for visual detection and CLAP for audio matching. "
               "Satellite scoring requires calibrated spatial measurements; plain RGB uploads remain traceable but incomplete.")
import_session_evidence(st.session_state)
with st.expander("Add satellite image"):
    from satellite_evidence import render_satellite_upload
    render_satellite_upload("report_satellite_upload")
with st.expander("Add calibrated satellite measurement"):
    from satellite_evidence import render_calibrated_satellite_measurement
    render_calibrated_satellite_measurement("report_calibrated_satellite")
with st.expander("Add multi-date Sentinel-2 SAFE archives"):
    from satellite_evidence import render_temporal_safe_archives
    render_temporal_safe_archives("report_temporal_satellite")

left, right = st.columns(2)
with left:
    project = saved_control(st.text_input, "report_project", "Project name", "Wildlife monitoring")
    zone = saved_control(st.text_input, "report_zone", "Zone / location", "")
with right:
    monitoring_date = saved_control(st.date_input, "report_date", "Monitoring date", date.today())
    st.caption("Project, location and date are your report labels. They do not establish when or where each recording was captured.")
notes = saved_control(st.text_area, "report_notes", "Your observations / report notes", "",
                      help="Describe the recording context or observations. These notes are kept separate from model findings.")
human_review_confirmed = st.checkbox("I have reviewed the source evidence, model outputs, limitations, and retrieved documents.",
                                     key="report_human_review")
history_text = saved_control(st.text_area, "report_historical_scores", "Historical fusion scores (optional)", "",
                             help="Comma-separated 0–100 scores from comparable prior observations. At least three are required for anomaly assessment.")
try:
    historical_scores = parse_historical_scores(history_text)
except ValueError as exc:
    historical_scores = []
    st.warning(str(exc))
with st.expander("Fusion configuration"):
    st.caption("Prototype decision-level weights. They are renormalized across available modalities and stored in the report.")
    weight_cols = st.columns(3)
    fusion_weights = {}
    for column, modality in zip(weight_cols, ("camera", "audio", "satellite")):
        with column:
            fusion_weights[modality] = st.slider(f"{modality.title()} weight", min_value=0.0, max_value=1.0,
                                                  value=float(FUSION_WEIGHTS[modality]), step=0.05,
                                                  key=f"report_weight_{modality}")
with st.expander("Groq report model"):
    groq_models = available_groq_models()
    if groq_models:
        groq_model = saved_control(st.selectbox, "report_groq_model", "Available models for this Groq key", groq_models,
                                   help="The list is fetched from your Groq account. This report uses the Chat Completions API; selecting a non-chat model leaves the local fallback active.")
        st.caption("The selected model applies to this report session only; it is not saved to source code.")
    else:
        groq_model = None
        st.caption("Set GROQ_API_KEY to load the models available to your account. GROQ_MODEL remains supported for non-interactive deployments.")
with st.expander("Add supporting monitoring documents"):
    references = st.file_uploader("Upload UTF-8 TXT or Markdown documents", type=["txt", "md"], accept_multiple_files=True,
                                  help="Documents are retrieved locally in this browser session and cited in the report. They do not change sensor evidence.")
    documents = []
    for reference in references or []:
        try:
            documents.append(ingest_document(reference.getvalue(), reference.name))
        except ValueError as exc:
            st.error(f"{reference.name}: {exc}")
    st.session_state["mrv_supporting_documents"] = documents

records = list(st.session_state.get("wildlife_evidence", {}).values())
if not records:
    st.info("No evidence yet. Upload audio in Bioacoustics or analyze an image, video or camera capture in Live Camera & Video, then return here.")
    st.page_link("bioacoustics.py", label="Open Bioacoustics", icon=":material/graphic_eq:")
    st.page_link("live_camera.py", label="Open Live Camera & Video", icon=":material/videocam:")
else:
    lookup = {r["id"]: r for r in records}
    selected = st.multiselect("Evidence to include", list(lookup), default=list(lookup),
                              format_func=lambda key: f"{KINDS[lookup[key]['kind']]} — {lookup[key]['source']} ({lookup[key]['identity'][:8]})")
    included = [lookup[key] for key in selected]
    report = build_report(included, project, zone, monitoring_date, notes, historical_scores=historical_scores,
                          fusion_weights=fusion_weights, human_review_confirmed=human_review_confirmed, groq_model=groq_model)
    query = " ".join([project, zone, notes, " ".join(report["possible_animal_labels"]), "environmental monitoring evidence"])
    retrieved = retrieve(st.session_state.get("mrv_supporting_documents", []), query)
    report = build_report(included, project, zone, monitoring_date, notes, retrieved_documents=retrieved,
                          historical_scores=historical_scores, fusion_weights=fusion_weights,
                          human_review_confirmed=human_review_confirmed, groq_model=groq_model)
    # Older loaded report builders may omit this newer field. Derive it from
    # selected evidence so both the card and JSON export stay accurate.
    report["satellite_count"] = sum(record.get("kind") == "satellite" for record in included)
    st.caption("The report updates from the selected saved results; opening it does not rerun animal detection.")
    audio_col, video_col, camera_col, satellite_col = st.columns(4)
    for column, label, kinds in [(audio_col, "Audio", {"audio"}), (video_col, "Video", {"video"}),
                                  (camera_col, "Images & camera", {"image", "camera_photo", "live_camera"})]:
        subset = [r for r in included if r["kind"] in kinds]
        with column:
            with st.container(border=True):
                st.subheader(label)
                st.metric("Evidence items", len(subset))
                st.write(f"Completed animal analyses: {sum(r['analysis'] is not None for r in subset)}")
                labels = sorted({a["animal"] for r in subset for a in (r.get("analysis") or {}).get("animals", [])})
                st.write("Possible animals: " + (", ".join(labels) if labels else "No matches available"))
    with satellite_col:
        with st.container(border=True):
            st.subheader("Satellite")
            st.metric("Images included", report["satellite_count"])
            temporal_records = [r for r in included if (r.get("measurements") or {}).get("temporal_ndvi_series")]
            st.caption("Temporal B04/B08 analysis included." if temporal_records else "File details inspected. Vegetation/change analysis unavailable.")
    evidence_object = report["ecomrv_evidence"]
    st.subheader("Multi-modal ecosystem evidence")
    st.caption("Project-specific indicators derived only from the selected analysed evidence. They are not certified ecological metrics.")
    fusion, agreement, anomaly = evidence_object["fusion"], evidence_object["agreement"], evidence_object["anomaly"]
    score_cols = st.columns(4)
    score_cols[0].metric("Fusion score", "Insufficient evidence" if fusion["fusion_score"] is None else f"{fusion['fusion_score']:.1f}/100")
    score_cols[1].metric("Agreement", "Insufficient evidence" if agreement["agreement_score"] is None else f"{agreement['agreement_score']:.1f}/100")
    score_cols[2].metric("Anomaly", "Historical data needed" if anomaly["anomaly_score"] is None else f"{anomaly['anomaly_score']:.1f}/100")
    score_cols[3].metric("Proposed MEHS", "Insufficient evidence" if evidence_object["mehs"] is None else f"{evidence_object['mehs']:.1f}/100")
    if evidence_object["missing_data_flags"]:
        st.warning("Incomplete evidence: " + ", ".join(evidence_object["missing_data_flags"]) + ". Missing inputs are excluded rather than treated as zero.")
    st.caption(agreement["explanation"])
    with st.expander("Evidence traceability and scoring details"):
        st.json(evidence_object)
    if retrieved:
        st.subheader("Retrieved supporting documentation")
        st.caption("These excerpts are source material for human review; they do not add observations or measurements.")
        for source in retrieved:
            with st.expander(f"{source['source']} — excerpt {source['chunk_index'] + 1}"):
                st.write(source["excerpt"])
    st.subheader("Evidence-grounded explanation")
    explanation = report["generated_explanation"]
    st.caption(explanation["generator"])
    st.markdown("**Observed data**")
    for item in explanation["observed_data"]:
        st.write(f"• {item}")
    st.markdown("**Derived-score interpretation**")
    for item in explanation["derived_score_interpretation"]:
        st.write(f"• {item}")
    if explanation.get("llm_interpretation"):
        st.markdown("**Groq-generated evidence interpretation**")
        st.write(explanation["llm_interpretation"])
    st.warning("Human verification is required. The explanation does not confirm species, environmental damage, or illegal activity.")
    if report["human_review"]["confirmed"]:
        st.success("Review gate recorded: human review confirmed for this report session.")
    else:
        st.info("Review gate: further verification required before using this report for operational decisions.")
    st.subheader("Final analysis")
    st.write(report["summary"])
    st.subheader("Animal findings")
    if report["findings"]:
        st.dataframe(report["findings"], hide_index=True)
    else:
        st.info("No animal findings in the selected evidence. Pending and uncertain results are explained below.")

    # ── Satellite & Earth Observation Findings Table ──
    if report.get("satellite_findings"):
        st.subheader("🛰️ Satellite & Earth Observation Evidence")
        st.caption("Temporal Sentinel-2 Level-2A B04/B08 NDVI vegetation canopy metrics from uploaded evidence.")
        st.dataframe(report["satellite_findings"], hide_index=True)

    # ── Three-Feature Modality Status ──
    st.subheader("🌐 Three-Feature Modality Status")
    features_meta = report.get("feature_summaries", {})
    f_cols = st.columns(3)
    for col, (f_key, f_icon) in zip(f_cols, [("camera", "📷"), ("audio", "🎙️"), ("satellite", "🛰️")]):
        meta = features_meta.get(f_key, {})
        with col:
            with st.container(border=True):
                st.markdown(f"**{f_icon} {meta.get('name', f_key.title())}**")
                status_color = "🟢" if meta.get("status") == "Available" else "⚪"
                st.caption(f"{status_color} Status: **{meta.get('status', 'Missing')}**")
                st.write(meta.get("details", ""))

    st.subheader("Cross-source analysis")
    st.write(report["cross_source_analysis"])
    st.subheader("Detected events and review items")
    if report["events"]:
        for event in report["events"]:
            st.write(f"• {event}")
    else:
        st.caption("No additional review events were produced by the selected analyses.")
    st.subheader("Evidence details")
    for record in included:
        with st.expander(f"{KINDS[record['kind']]} — {record['source']}"):
            st.caption(f"Evidence ID: {record['id']}")
            if record["kind"] == "satellite":
                preview = st.session_state.get("satellite_previews", {}).get(record["identity"])
                if preview:
                    st.image(preview, caption=record["source"])
            if record.get("measurements"):
                st.write("Measurements / inspected file details")
                st.json(record["measurements"])
            if record.get("analysis") is not None:
                st.json(record["analysis"])
            else:
                st.info("Vegetation and change analysis unavailable." if record["kind"] == "satellite" else "Animal identification has not been completed for this media item.")
    st.subheader("Recommendations")
    for recommendation in report["recommendations"]:
        st.write(f"• {recommendation}")
    if notes.strip():
        st.subheader("Your observations")
        st.text(notes)
        st.caption("User-supplied notes; not independently verified by the analysis.")
    with st.expander("Interpretation and limitations"):
        for limitation in report["limitations"]:
            st.write(f"• {limitation}")
    st.download_button("Download report (Markdown)", markdown_report(report),
                       file_name="wildlife_analysis_report.md", mime="text/markdown")
    st.download_button("Download report with evidence (JSON)", json.dumps(report, indent=2),
                       file_name="wildlife_analysis_report.json", mime="application/json")
