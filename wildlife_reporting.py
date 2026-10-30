"""Session evidence and report assembly. No model inference or demo data."""
import copy
import hashlib
import json
from datetime import datetime, timezone

from ecosystem_pipeline import build_evidence_object
from report_explanation import generate_explanation

KINDS = {"audio": "Audio", "image": "Image", "video": "Video", "camera_photo": "Camera photo", "live_camera": "Live camera", "satellite": "Satellite image"}


def save_evidence(state, kind, identity, source, analysis=None, measurements=None):
    if kind not in KINDS:
        raise ValueError("Unsupported evidence type")
    key = f"{kind}:{identity}"
    records = dict(state.get("wildlife_evidence", {}))
    record = copy.deepcopy(records.get(key, {
        "id": key, "kind": kind, "source": source, "identity": identity,
        "added_at": datetime.now(timezone.utc).isoformat(),
        "analysis": None, "measurements": None,
    }))
    record["source"] = source
    if analysis is not None:
        record["analysis"] = copy.deepcopy(analysis)
    if measurements is not None:
        record["measurements"] = copy.deepcopy(measurements)
    records[key] = record
    state["wildlife_evidence"] = records
    return key


def import_session_evidence(state):
    """Include results made before opening Reports or before this feature existed."""
    audio = state.get("audio_animal_result")
    if audio:
        save_evidence(state, "audio", audio[0], audio[1].get("source", "Audio recording"), analysis=audio[1])
    visual = state.get("wildlife_result")
    if visual:
        report = visual[1]
        kind = report.get("media_type", "video" if "duration_seconds" in report else "image")
        save_evidence(state, kind, report.get("media_sha256", visual[0]), report.get("source", "Visual media"), analysis=report)
    live = state.get("live_animal_report")
    if live:
        save_evidence(state, "live_camera", live.get("session_id", "previous-live-session"), "Browser webcam", analysis=live)


def build_report(records, project, zone, monitoring_date, notes, retrieved_documents=None, historical_scores=None, fusion_weights=None,
                 human_review_confirmed=False, groq_model=None):
    records = copy.deepcopy(records)
    rows, events, audio_names, visual_names = [], [], set(), set()
    analyzed = 0
    satellite_count = sum(r["kind"] == "satellite" for r in records)
    for record in records:
        if record["kind"] == "satellite":
            temporal = (record.get("measurements") or {}).get("temporal_features")
            if temporal:
                events.append(f"{record['source']}: Sentinel-2 temporal NDVI evidence included; vegetation-change signals require review and are not a deforestation determination.")
            else:
                events.append(f"{record['source']}: satellite image included as evidence; vegetation and change analysis are unavailable.")
            continue
        analysis = record.get("analysis")
        if analysis is None:
            events.append(f"{record['source']}: animal identification has not been completed.")
            continue
        analyzed += 1
        is_audio = record["kind"] == "audio"
        names = audio_names if is_audio else visual_names
        for animal in analysis.get("animals", []):
            name = animal["animal"]
            names.add(name.strip().lower())
            if is_audio:
                intervals = animal.get("intervals", [])
                timestamps = "; ".join(f"{a:.2f}–{b:.2f}s" for a, b in intervals)
                score = animal.get("max_similarity")
                count = animal.get("windows_matched", 0)
            else:
                timestamps = f"{animal.get('first_seen_seconds', 0):.2f}–{animal.get('last_seen_seconds', 0):.2f}s"
                score = animal.get("max_score")
                count = animal.get("observations", 0)
            rows.append({"animal": name, "evidence": KINDS[record["kind"]], "source": record["source"],
                         "evidence_id": record["id"], "score": score,
                         "score_type": "Audio cosine similarity" if is_audio else "Visual matching score",
                         "observations": count, "time_ranges": timestamps,
                         "call_description": animal.get("call_description", "Not applicable")})
        if is_audio:
            windows = analysis.get("windows", [])
            uncertain = sum(w.get("status") == "uncertain" for w in windows)
            silent = sum(w.get("status") == "silence" for w in windows)
            background = sorted({w["closest_match"] for w in windows if w.get("status") == "background"})
            if uncertain:
                events.append(f"{record['source']}: {uncertain} uncertain audio window(s) need listening review.")
            if silent:
                events.append(f"{record['source']}: {silent} silent/near-silent window(s).")
            if background:
                events.append(f"{record['source']}: background sound matches — {', '.join(background)}.")
        if not analysis.get("animals"):
            events.append(f"{record['source']}: no animal match passed the analysis criteria; this does not establish absence.")
    shared = sorted(audio_names & visual_names)
    if audio_names and visual_names:
        comparison = (f"Shared predicted labels across audio and visual evidence: {', '.join(shared)}. " if shared else
                      "No identical predicted labels occur in both the audio and visual results. ")
        comparison += "The uploads are not time/location aligned; label overlap alone does not confirm the same animal or event."
    else:
        comparison = "Cross-source comparison is incomplete: animal matches are needed from both audio and visual evidence."
    unique_names = sorted(audio_names | visual_names)
    if not records:
        summary = "No user evidence is available. Upload and analyze audio, images, video, or camera captures to produce an assessment."
    else:
        summary = f"This report contains {len(records)} evidence item(s), with animal analysis completed for {analyzed}. "
        summary += (f"Possible animal labels reported: {', '.join(unique_names)}. " if unique_names else
                    "No animal labels have passed the available analysis criteria. ")
        summary += "These are model predictions for review, not confirmed species counts or an assessment of ecosystem health."
    if satellite_count:
        temporal_count = sum(bool((r.get("measurements") or {}).get("temporal_features")) for r in records if r["kind"] == "satellite")
        summary += (f" {temporal_count} calibrated Sentinel-2 temporal evidence set(s) are included; findings remain subject to spatial-alignment and seasonal limitations." if temporal_count else
                    f" {satellite_count} satellite image(s) are included with file details; vegetation condition and change have not been assessed.")
        comparison += " Satellite evidence is not automatically time/location aligned with wildlife media."

    # Structured satellite findings table across observation dates
    satellite_findings = []
    for record in records:
        if record["kind"] == "satellite":
            meas = record.get("measurements") or {}
            series = meas.get("temporal_ndvi_series") or []
            if series:
                for s in series:
                    satellite_findings.append({
                        "source": record["source"],
                        "observation_date": s.get("acquisition_date", "Unknown"),
                        "tile_id": s.get("tile_id", "N/A"),
                        "mean_ndvi": f"{s.get('mean_ndvi', 0.0):.3f}" if s.get("mean_ndvi") is not None else "N/A",
                        "median_ndvi": f"{s.get('median_ndvi', 0.0):.3f}" if s.get("median_ndvi") is not None else "N/A",
                        "std_ndvi": f"{s.get('std_ndvi', 0.0):.3f}" if s.get("std_ndvi") is not None else "N/A",
                        "valid_pixels": f"{s.get('valid_pixels', 0):,}",
                        "notes": "Sentinel-2 L2A B04/B08 temporal NDVI"
                    })
            elif meas.get("ndvi") is not None:
                satellite_findings.append({
                    "source": record["source"],
                    "observation_date": meas.get("observation_date", "Single date"),
                    "tile_id": "Calibrated",
                    "mean_ndvi": f"{float(meas['ndvi']):.3f}",
                    "median_ndvi": "N/A",
                    "std_ndvi": "N/A",
                    "valid_pixels": "Calibrated",
                    "notes": f"Cloud cover: {meas.get('cloud_cover', 0)}%"
                })

    # Three-feature status summary
    v_items = sum(r["kind"] in {"image", "video", "camera_photo", "live_camera"} for r in records)
    a_items = sum(r["kind"] == "audio" for r in records)
    feature_summaries = {
        "camera": {
            "name": "Camera & Visual",
            "status": "Available" if (visual_names or v_items) else "Missing",
            "items_count": v_items,
            "detected_species": sorted(visual_names),
            "details": f"{v_items} visual item(s) uploaded. Detected classes: {', '.join(sorted(visual_names)) or 'None'}."
        },
        "audio": {
            "name": "Audio & Bioacoustics",
            "status": "Available" if (audio_names or a_items) else "Missing",
            "items_count": a_items,
            "detected_species": sorted(audio_names),
            "details": f"{a_items} audio recording(s) uploaded. Matched calls: {', '.join(sorted(audio_names)) or 'None'}."
        },
        "satellite": {
            "name": "Satellite Earth Observation",
            "status": "Available" if satellite_count else "Missing",
            "items_count": satellite_count,
            "details": f"{satellite_count} dataset(s). {len(satellite_findings)} observation date(s) tracked." if satellite_count else "No satellite imagery or NDVI data uploaded."
        }
    }

    evidence_object = build_evidence_object(records, project.strip(), zone.strip(), history=historical_scores,
                                             fusion_weights=fusion_weights)
    explanation = generate_explanation(evidence_object, list(retrieved_documents or []), groq_model=groq_model)
    return {
        "title": "EcoMRV Wildlife Evidence Report", "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": project.strip(), "zone": zone.strip(), "monitoring_date": str(monitoring_date),
        "user_notes": notes.strip(), "metadata_provenance": "Project, zone, date and notes are user supplied; not extracted from media.",
        "summary": summary, "evidence_count": len(records), "analyzed_count": analyzed,
        "satellite_count": satellite_count,
        "possible_animal_labels": unique_names, "findings": rows, "events": events,
        "satellite_findings": satellite_findings,
        "feature_summaries": feature_summaries,
        "cross_source_analysis": comparison, "shared_labels": shared,
        "recommendations": ["Review the source frames and replay matching audio intervals before confirming species.",
                            "Review uncertain results and collect clearer recordings where needed.",
                            "Record capture time and location for each source before comparing events across devices."],
        "limitations": ["Audio and visual scores use different scales and are not averaged or treated as accuracy percentages.",
                        "Repeated observations and matching windows do not count unique animals.",
                        "Video uses sampled frames; live camera reports retain the latest 300 analyzed frames.",
                        "Ecosystem health, environmental risk and causal relationships cannot be determined from these media results alone."],
        "evidence": records, "ecomrv_evidence": evidence_object,
        "retrieved_documents": list(retrieved_documents or []),
        "generated_explanation": explanation,
        "human_review": {"required": True, "confirmed": bool(human_review_confirmed),
                         "status": "Human review confirmed" if human_review_confirmed else "Further verification required"},
    }


def markdown_report(report):
    lines = [f"# {report['title']}", "", f"Project: {report['project'] or 'Not supplied'}",
             f"Zone: {report['zone'] or 'Not supplied'}", f"Monitoring date: {report['monitoring_date']}",
             f"Generated: {report['generated_at']}", "", report["metadata_provenance"],
             "", "## Final analysis", "", report["summary"], "", "## User notes", "", report["user_notes"] or "None",
             "", "## Animal findings", ""]
    for row in report["findings"]:
        lines.append(f"- {row['animal']} — {row['evidence']} / {row['source']}; {row['score_type']}: {row['score']}; "
                     f"observations: {row['observations']}; time: {row['time_ranges']}; evidence: {row['evidence_id']}")
    if not report["findings"]:
        lines.append("No animal findings available.")

    # Only include Satellite & vegetation section if satellite evidence is present
    sat_findings = report.get("satellite_findings", [])
    if sat_findings and report.get("satellite_count", 0) > 0:
        lines.extend(["", "## Satellite and vegetation evidence", ""])
        for s in sat_findings:
            lines.append(f"- {s['source']} — Date: {s['observation_date']}; Tile: {s['tile_id']}; Mean NDVI: {s['mean_ndvi']}; Valid pixels: {s['valid_pixels']}; {s['notes']}")

    lines.extend(["", "## Cross-source analysis", "", report["cross_source_analysis"], "", "## Events and review items", ""])
    lines.extend(f"- {event}" for event in report["events"])
    lines.extend(["", "## Evidence and coverage", ""])
    for record in report["evidence"]:
        analysis = record.get("analysis") or {}
        lines.append(f"- {record['source']} ({KINDS[record['kind']]}); ID: {record['id']}; "
                     f"model: {analysis.get('model', 'Identification not run')}; "
                     f"coverage: {analysis.get('coverage', str(analysis.get('duration_seconds', 'Single frame')) + ' seconds' if 'duration_seconds' in analysis else 'Single frame or not available')}")
        if record.get("measurements"):
            lines.append(f"  Measurements / inspected file details: {json.dumps(record['measurements'], ensure_ascii=False)}")
    ecomrv = report.get("ecomrv_evidence")
    if ecomrv:
        fusion, agreement, anomaly = ecomrv["fusion"], ecomrv["agreement"], ecomrv["anomaly"]
        lines.extend(["", "## Multi-modal evidence indicators", "",
                      f"Available modalities: {', '.join(fusion['available_modalities']) or 'None'}",
                      f"Missing/incomplete modalities: {', '.join(fusion['missing_modalities']) or 'None'}",
                      f"Fusion score: {fusion['fusion_score'] if fusion['fusion_score'] is not None else 'Insufficient evidence'}",
                      f"Agreement score: {agreement['agreement_score'] if agreement['agreement_score'] is not None else 'Insufficient evidence'} ({agreement['agreement_level']})",
                      f"Anomaly score: {anomaly['anomaly_score'] if anomaly['anomaly_score'] is not None else 'Further historical data required'}",
                      f"{ecomrv['label']}: {ecomrv['mehs'] if ecomrv['mehs'] is not None else 'Insufficient evidence'}",
                      "Human verification is required before operational decisions."])
    if report.get("retrieved_documents"):
        lines.extend(["", "## Retrieved supporting documentation", ""])
        for source in report["retrieved_documents"]:
            lines.append(f"- {source['source']} (chunk {source['chunk_index']}, retrieval score {source['score']}): {source['excerpt']}")
    explanation = report.get("generated_explanation")
    if explanation:
        lines.extend(["", "## Evidence-grounded explanation", "", f"Generator: {explanation['generator']}", "", "### Observed data"])
        lines.extend(f"- {item}" for item in explanation["observed_data"])
        lines.extend(["", "### Derived-score interpretation"])
        lines.extend(f"- {item}" for item in explanation["derived_score_interpretation"])
        if explanation.get("llm_interpretation"):
            lines.extend(["", "### LLM Evidence Interpretation", "", explanation["llm_interpretation"]])
        lines.extend(["", "### Human review", "", "Human verification is required before operational decisions."])
    review = report.get("human_review")
    if review:
        lines.extend(["", "## Review gate", "", f"Status: {review['status']}"])
    for title, key in [("Recommendations", "recommendations"), ("Limitations", "limitations")]:
        lines.extend(["", f"## {title}", ""])
        lines.extend(f"- {item}" for item in report[key])
    return "\n".join(lines)
