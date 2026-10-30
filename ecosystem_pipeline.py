"""Traceable, missing-data-aware multi-modal evidence processing.

This module only derives project-specific indicators from analysed evidence.
It never interprets an ordinary RGB satellite upload as NDVI data.
"""
from __future__ import annotations

from datetime import datetime, timezone
from math import sqrt
from statistics import median
from typing import Any

from ecomrv_config import AGREEMENT_LEVELS, FUSION_WEIGHTS, MEHS_WEIGHTS


def _clamp(value: float) -> float:
    return round(max(0.0, min(100.0, float(value))), 2)


def parse_historical_scores(text: str) -> list[float]:
    """Parse user-declared prior fusion scores for transparent anomaly comparison."""
    if not text.strip():
        return []
    try:
        values = [float(value.strip()) for value in text.replace("\n", ",").split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("Historical scores must be comma-separated numbers from 0 to 100.") from exc
    if len(values) < 3:
        raise ValueError("Provide at least three historical scores for anomaly assessment.")
    if any(not 0 <= value <= 100 for value in values):
        raise ValueError("Historical scores must be between 0 and 100.")
    return values


def visual_features(analysis: dict[str, Any]) -> dict[str, Any]:
    animals = analysis.get("animals", [])
    scores = [float(row.get("max_score", 0.0)) for row in animals if row.get("max_score") is not None]
    return {
        "modality": "camera", "raw": {"detected_classes": [row.get("animal") for row in animals],
        "wildlife_observations": sum(int(row.get("observations", 0)) for row in animals),
        "species_count": len(animals), "max_detection_score": max(scores, default=None),
        "bounding_boxes": [d.get("box") for o in analysis.get("observations", []) for d in o.get("detections", [])],
        "timestamp": analysis.get("created_at"), "model_version": analysis.get("model")},
        "normalized_score": _clamp(max(scores, default=0.0) * 100) if scores else None,
        "status": "available" if scores else "insufficient_evidence",
    }


def audio_features(analysis: dict[str, Any]) -> dict[str, Any]:
    animals, windows = analysis.get("animals", []), analysis.get("windows", [])
    scores = [float(row.get("max_similarity", 0.0)) for row in animals if row.get("max_similarity") is not None]
    active = sum(row.get("status") == "possible_animal" for row in windows)
    # CLAP cosine similarity is mapped only for a transparent prototype signal:
    # its configured acceptance floor maps to zero, and 1.0 maps to 100.
    score = _clamp((max(scores) - 0.25) / 0.75 * 100) if scores else None
    return {
        "modality": "audio", "raw": {"detected_classes": [row.get("animal") for row in animals],
        "species_count": len(animals), "acoustic_activity_windows": active,
        "window_count": len(windows), "max_similarity": max(scores, default=None),
        "timestamp": analysis.get("created_at"), "model_version": analysis.get("model")},
        "normalized_score": score, "status": "available" if score is not None else "insufficient_evidence",
    }


def satellite_features(measurements: dict[str, Any] | None) -> dict[str, Any]:
    measurements = measurements or {}
    temporal = measurements.get("temporal_features") or {}
    # Try to get NDVI from temporal analysis result, fallback to direct ndvi field
    ndvi = temporal.get("latest_ndvi") or temporal.get("mean_ndvi") or measurements.get("ndvi")
    previous = measurements.get("previous_ndvi")
    # For temporal analysis, use the overall mean change if available
    mean_change = temporal.get("mean_change")
    # Existing upload inspection has no calibrated bands. It remains traceable,
    # but is deliberately excluded from numeric fusion.
    if not isinstance(ndvi, (int, float)):
        return {"modality": "satellite", "raw": measurements, "normalized_score": None,
                "status": "insufficient_evidence", "reason": "No comparable calibrated NDVI measurement is available."}
    # Use mean_change from temporal analysis if available, otherwise calculate from previous
    change = mean_change if isinstance(mean_change, (int, float)) else (float(ndvi) - float(previous) if isinstance(previous, (int, float)) else None)
    return {"modality": "satellite", "raw": {**measurements, "ndvi_change": change, "temporal": temporal},
            "normalized_score": _clamp(float(ndvi) * 100), "status": "available"}


def weighted_fusion(features: dict[str, dict[str, Any]], weights: dict[str, float] | None = None) -> dict[str, Any]:
    weights = dict(weights or FUSION_WEIGHTS)
    available = [name for name, item in features.items() if item.get("normalized_score") is not None]
    missing = [name for name in weights if name not in available]
    if not available:
        return {"fusion_score": None, "weights": {}, "contributions": {}, "available_modalities": [],
                "missing_modalities": missing, "confidence": "insufficient evidence"}
    total = sum(max(0.0, weights.get(name, 0.0)) for name in available)
    effective = {name: max(0.0, weights.get(name, 0.0)) / total for name in available} if total else {name: 1 / len(available) for name in available}
    contributions = {name: round(features[name]["normalized_score"] * effective[name], 2) for name in available}
    return {"fusion_score": round(sum(contributions.values()), 2), "weights": effective, "contributions": contributions,
            "available_modalities": available, "missing_modalities": missing,
            "confidence": "complete" if not missing else "incomplete evidence"}


def agreement_score(features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = {name: item["normalized_score"] for name, item in features.items() if item.get("normalized_score") is not None}
    if len(values) < 2:
        return {"agreement_score": None, "agreement_level": "insufficient evidence", "modality_values": values,
                "explanation": "At least two available modalities are required to assess cross-modal agreement."}
    mean = sum(values.values()) / len(values)
    std = sqrt(sum((value - mean) ** 2 for value in values.values()) / len(values))
    score = _clamp(100 - 2 * std)
    level = next(label for floor, label in AGREEMENT_LEVELS if score >= floor)
    return {"agreement_score": score, "agreement_level": level, "modality_values": values,
            "explanation": "Agreement is 100 minus twice the population standard deviation of available normalized modality scores."}


def anomaly_score(current: float | None, history: list[float] | None = None) -> dict[str, Any]:
    history = [float(v) for v in (history or []) if isinstance(v, (int, float))]
    if current is None or len(history) < 3:
        return {"anomaly_score": None, "is_anomaly": None, "method": "unavailable", "historical_baseline": history or None,
                "current_value": current, "explanation": "Further historical observations are required for anomaly assessment."}
    baseline = median(history)
    mad = median([abs(value - baseline) for value in history])
    scale = max(1.4826 * mad, 1.0)
    z = abs(float(current) - baseline) / scale
    score = _clamp(z / 3 * 100)
    return {"anomaly_score": score, "is_anomaly": z >= 3, "method": "robust median/MAD deviation",
            "historical_baseline": baseline, "current_value": current,
            "explanation": "An anomaly indicates deviation from the supplied historical pattern, not environmental damage."}


def ecosystem_and_mehs(fusion: dict[str, Any], agreement: dict[str, Any], anomaly: dict[str, Any]) -> dict[str, Any]:
    ecosystem = fusion.get("fusion_score")
    if ecosystem is None:
        return {"ecosystem_score": None, "mehs": None, "label": "Insufficient evidence", "components": {}}
    agree = agreement.get("agreement_score")
    risk = anomaly.get("anomaly_score")
    # Missing agreement/anomaly are neutral but reported as unavailable, not zero.
    mehs = ecosystem * MEHS_WEIGHTS["fusion"] + (agree if agree is not None else 50) * MEHS_WEIGHTS["agreement"] - (risk if risk is not None else 0) * MEHS_WEIGHTS["anomaly_risk"]
    return {"ecosystem_score": round(ecosystem, 2), "mehs": _clamp(mehs),
            "label": "Proposed Multi-Modal Ecosystem Health Score", "components": {"fusion": ecosystem, "agreement": agree, "anomaly_risk": risk, "weights": MEHS_WEIGHTS}}


def build_evidence_object(records: list[dict[str, Any]], project: str, zone: str, history: list[float] | None = None,
                          fusion_weights: dict[str, float] | None = None) -> dict[str, Any]:
    visuals = [r.get("analysis") for r in records if r.get("kind") in {"image", "video", "camera_photo", "live_camera"} and r.get("analysis")]
    audios = [r.get("analysis") for r in records if r.get("kind") == "audio" and r.get("analysis")]
    satellites = [r.get("measurements") for r in records if r.get("kind") == "satellite"]
    features = {"camera": visual_features(visuals[-1]) if visuals else {"modality": "camera", "normalized_score": None, "status": "missing"},
                "audio": audio_features(audios[-1]) if audios else {"modality": "audio", "normalized_score": None, "status": "missing"},
                "satellite": satellite_features(satellites[-1] if satellites else None)}
    fusion = weighted_fusion(features, fusion_weights)
    agreement = agreement_score(features)
    anomaly = anomaly_score(fusion["fusion_score"], history)
    scores = ecosystem_and_mehs(fusion, agreement, anomaly)
    return {"schema_version": "1.0", "project": project, "zone": zone, "processing_timestamp": datetime.now(timezone.utc).isoformat(),
            "features": features, "normalized_scores": {k: v.get("normalized_score") for k, v in features.items()},
            "fusion": fusion, "agreement": agreement, "anomaly": anomaly, **scores,
            "missing_data_flags": fusion["missing_modalities"], "human_review_required": True,
            "limitations": ["MEHS is a project-specific evidence indicator, not a certified ecological metric.",
                            "Satellite evidence is excluded until calibrated spatial vegetation measurements are supplied."]}
