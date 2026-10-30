"""Inspect uploaded satellite images without inventing spectral measurements."""
import hashlib
import io
import json
import warnings
from datetime import datetime, timezone

from PIL import Image

from wildlife_reporting import save_evidence


def inspect_satellite(data):
    if len(data) > 50 * 1024 * 1024:
        raise ValueError("Use an image smaller than 50 MB.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as image:
            measurements = {
                "width_pixels": image.width, "height_pixels": image.height,
                "format": image.format, "pixel_mode": image.mode,
                "file_size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "inspection_status": "Image decoded and file details inspected; no spectral analysis performed.",
                "vegetation_analysis": "Unavailable: calibrated red/NIR bands and their mapping have not been supplied to an analysis pipeline.",
                "change_analysis": "Unavailable: no aligned before/after analysis has been performed.",
            }
            image.thumbnail((900, 900))
            preview = io.BytesIO()
            image.convert("RGB").save(preview, format="PNG")
    return measurements, preview.getvalue()


def calibrated_spatial_measurements(ndvi, previous_ndvi=None, observation_date=None, geometry=None, cloud_cover_pct=None):
    """Validate declared calibrated satellite metrics without deriving them from RGB pixels."""
    ndvi = float(ndvi)
    if not -1.0 <= ndvi <= 1.0:
        raise ValueError("NDVI must be between -1 and 1.")
    if previous_ndvi is not None:
        previous_ndvi = float(previous_ndvi)
        if not -1.0 <= previous_ndvi <= 1.0:
            raise ValueError("Previous NDVI must be between -1 and 1.")
    if cloud_cover_pct is not None and not 0 <= float(cloud_cover_pct) <= 100:
        raise ValueError("Cloud cover must be between 0 and 100 percent.")
    change = ndvi - previous_ndvi if previous_ndvi is not None else None
    return {"ndvi": ndvi, "previous_ndvi": previous_ndvi, "ndvi_change": change,
            "vegetation_change": "decline" if change is not None and change < -0.05 else "increase" if change is not None and change > 0.05 else "stable/undetermined",
            "observation_date": str(observation_date) if observation_date else None, "geometry": geometry or None,
            "cloud_cover_pct": float(cloud_cover_pct) if cloud_cover_pct is not None else None,
            "measurement_provenance": "User-supplied calibrated satellite measurement; not calculated from this application\'s RGB upload.",
            "processing_timestamp": datetime.now(timezone.utc).isoformat()}


def render_satellite_upload(key):
    import streamlit as st

    clip = st.file_uploader("Upload satellite image for Reports", type=["png", "jpg", "jpeg", "tif", "tiff"], key=key)
    st.caption("Adds your image and inspected file details to the combined report. Vegetation and change estimates require a separate spectral analysis.")
    if clip is None:
        return
    try:
        measurements, preview = inspect_satellite(clip.getvalue())
        save_evidence(st.session_state, "satellite", measurements["sha256"], clip.name, measurements=measurements)
        previews = dict(st.session_state.get("satellite_previews", {}))
        previews[measurements["sha256"]] = preview
        st.session_state["satellite_previews"] = previews
        st.image(preview, caption=clip.name)
        st.success("Satellite image added to Reports.")
    except Exception as exc:
        st.error(f"Cannot inspect this satellite image: {exc}")


def render_calibrated_satellite_measurement(key):
    """Optional traceable input for metrics produced by a calibrated external pipeline."""
    import streamlit as st

    st.caption("Enter only NDVI metrics produced by a calibrated satellite workflow. These fields are user supplied and do not make an RGB upload spectral data.")
    left, right = st.columns(2)
    with left:
        ndvi = st.number_input("Calibrated NDVI", min_value=-1.0, max_value=1.0, value=0.5, step=0.01, key=f"{key}_ndvi")
        has_previous = st.checkbox("Include previous NDVI", key=f"{key}_has_previous")
        previous = st.number_input("Previous calibrated NDVI", min_value=-1.0, max_value=1.0, value=0.5, step=0.01, key=f"{key}_previous") if has_previous else None
    with right:
        observation = st.date_input("Satellite observation date", key=f"{key}_date")
        cloud_cover = st.number_input("Cloud cover (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, key=f"{key}_cloud")
        geometry = st.text_area("Coordinates / geometry (optional)", key=f"{key}_geometry", height=68)
    if st.button("Add calibrated satellite measurement", key=f"{key}_save"):
        try:
            measurements = calibrated_spatial_measurements(ndvi, previous, observation, geometry, cloud_cover)
            identity = hashlib.sha256(json.dumps(measurements, sort_keys=True).encode()).hexdigest()
            save_evidence(st.session_state, "satellite", identity, "Calibrated satellite measurement", measurements=measurements)
            st.success("Calibrated spatial measurement added to Reports.")
        except ValueError as exc:
            st.error(f"Cannot add measurement: {exc}")


def render_temporal_safe_archives(key, locations=None):
    """Discover and analyse real Sentinel-2 L2A SAFE archives (ZIP or folder) without browser uploads.

    SAFE products commonly exceed Streamlit's upload limit, so local discovery
    intentionally reads the original archives/folders in place and never copies them
    into session state.  The result (not raster arrays) is retained as evidence.
    """
    import streamlit as st
    from satellite_temporal import analyze_safe_archives, discover_safe_zips

    st.caption("Discovers local Sentinel-2 Level-2A SAFE archives (ZIP or folder) and calculates NDVI only from B04/B08. Project/configuration change thresholds are displayed with the results.")
    paths = discover_safe_zips(locations)
    if not paths:
        st.info("No Sentinel-2 L2A SAFE archives (ZIP or folder) were discovered in Downloads, data/satellite, or current directory.")
        return
    st.write(f"Discovered {len(paths)} SAFE archive(s) (ZIP or folder), ordered by metadata timestamp:")
    for path in paths:
        st.caption(f"{path.name} ({'ZIP' if path.is_file() else 'Folder'})")
    
    # Add sampling option for speed
    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption("Processing speed (higher = faster, less complete):")
    with col2:
        sample_blocks = st.selectbox("Sampling rate", [1, 2, 4, 8], index=2, key=f"{key}_sample",
                                      help="1=full processing (slow), 4=fast (recommended), 8=fastest")
    
    if st.button("Analyse discovered Sentinel-2 dates", key=f"{key}_analyse", type="primary"):
        try:
            with st.spinner(f"Reading B04/B08 JP2 bands and calculating windowed NDVI (sampling every {sample_blocks}th block)…"):
                result = analyze_safe_archives(paths, output_dir="data/satellite_outputs", sample_blocks=sample_blocks)
            identity = hashlib.sha256(json.dumps([scene["sha256"] for scene in result["scenes"]], sort_keys=True).encode()).hexdigest()
            save_evidence(st.session_state, "satellite", identity, f"{result['date_count']} Sentinel-2 L2A SAFE observations", measurements=result)
            st.session_state[f"{key}_result"] = result
            st.success("Temporal Sentinel-2 evidence added to Reports.")
        except Exception as exc:
            st.error(f"Sentinel-2 analysis could not be completed: {exc}")
    result = st.session_state.get(f"{key}_result")
    if result:
        st.dataframe(result["temporal_ndvi_series"], hide_index=True)
        st.line_chart(result["temporal_ndvi_series"], x="acquisition_date", y="mean_ndvi")
        st.caption("Change thresholds (project/configuration thresholds, not validated deforestation thresholds): " + json.dumps(result["thresholds"]))
        st.json({"changes": result["changes"], "temporal_anomaly": result["temporal_anomaly"], "limitations": result["limitations"]})
