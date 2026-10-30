"""Two-date, multispectral vegetation comparison for the Satellite Detection tab."""
from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import date, timedelta
from typing import Any

import numpy as np


MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_ANALYSIS_SIDE = 1400


def _band_labels(dataset: Any) -> list[str]:
    labels = []
    for index in range(1, dataset.count + 1):
        description = dataset.descriptions[index - 1]
        tags = dataset.tags(index)
        useful_tags = [f"{key}={value}" for key, value in tags.items()]
        label = description.strip() if description else ""
        if useful_tags:
            label = f"{label} ({', '.join(useful_tags)})" if label else ", ".join(useful_tags)
        labels.append(label or f"Band {index}")
    return labels


def _band_role(label: str) -> str | None:
    text = label.casefold()
    if re.search(r"(?<![a-z0-9])b0?4(?![a-z0-9])|\bred\b", text):
        return "red"
    if re.search(r"(?<![a-z0-9])b0?8a?(?![a-z0-9])|\bnir\b", text) or "near infrared" in text or "near-infrared" in text:
        return "nir"
    if re.search(r"\b(scl|scene classification|classification layer)\b", text):
        return "scl"
    if re.search(r"\b(b0?3|green)\b", text):
        return "green"
    if re.search(r"\b(b0?2|blue)\b", text):
        return "blue"
    return None


def inspect_upload(payload: bytes, filename: str) -> dict[str, Any]:
    """Return raster metadata and a small display preview, without calculating NDVI."""
    if not payload:
        raise ValueError("The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Each upload must be 100 MB or smaller.")

    try:
        import rasterio
        from rasterio.io import MemoryFile

        with MemoryFile(payload) as memfile, memfile.open() as ds:
            labels = _band_labels(ds)
            roles: dict[str, int] = {}
            for band, label in enumerate(labels, start=1):
                role = _band_role(label)
                if role and role not in roles:
                    roles[role] = band
            # Read a low-cost preview only; the full comparison runs on button click.
            preview = _raster_preview(ds, labels, roles)
            colorinterp = {getattr(item, "name", str(item)).casefold() for item in ds.colorinterp}
            rgb_only = (
                filename.casefold().endswith((".png", ".jpg", ".jpeg"))
                or ("red" in colorinterp and "green" in colorinterp and "blue" in colorinterp
                    and "nir" not in roles and (ds.count <= 3 or "alpha" in colorinterp))
            )
            return {
                "kind": "raster", "filename": filename, "labels": labels,
                "roles": roles, "count": ds.count, "width": ds.width,
                "height": ds.height, "crs": ds.crs.to_string() if ds.crs else None,
                "preview": preview, "fingerprint": hashlib.sha256(payload).hexdigest(),
                "rgb_only": rgb_only,
                "preview_type": "True-color preview" if all(roles.get(band) for band in ("red", "green", "blue")) else "Band-composite preview",
            }
    except Exception as raster_error:
        # Ordinary RGB images are previewable, but they do not carry NIR data.
        try:
            from PIL import Image, ImageOps

            with Image.open(io.BytesIO(payload)) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                original_width, original_height = image.size
                image.thumbnail((600, 600))
                return {
                    "kind": "rgb", "filename": filename, "labels": [], "roles": {},
                    "count": len(image.getbands()), "width": original_width,
                    "height": original_height, "crs": None,
                    "preview": np.asarray(image), "rgb_only": True,
                    "fingerprint": hashlib.sha256(payload).hexdigest(),
                    "raster_error": str(raster_error),
                }
        except Exception as image_error:
            raise ValueError(f"Cannot read this satellite file as a raster or image: {image_error}") from image_error


def _stretch(channel: np.ndarray) -> np.ndarray:
    channel = np.asarray(channel, dtype=np.float32)
    valid = np.isfinite(channel)
    if not valid.any():
        return np.zeros(channel.shape, dtype=np.uint8)
    low, high = np.percentile(channel[valid], [2, 98])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((channel - low) / (high - low), 0.0, 1.0)
    scaled[~valid] = 0.0
    return (scaled * 255).astype(np.uint8)


def _raster_preview(ds: Any, labels: list[str], roles: dict[str, int], size: int = 600) -> np.ndarray:
    from rasterio.enums import Resampling

    color_bands = [roles.get(role) for role in ("red", "green", "blue")]
    natural_color = all(color_bands)
    if not natural_color:
        color_bands = list(range(1, min(ds.count, 3) + 1))
    if len(color_bands) < 3:
        color_bands = [color_bands[0]] * 3
    scale = min(1.0, size / max(ds.width, ds.height))
    out_h, out_w = max(1, round(ds.height * scale)), max(1, round(ds.width * scale))
    channels = [ds.read(band, out_shape=(out_h, out_w), masked=True,
                        resampling=Resampling.bilinear).astype(np.float32).filled(np.nan)
                for band in color_bands[:3]]
    preview = np.dstack([_stretch(channel) for channel in channels])
    return preview


def _selected_band(info: dict[str, Any], role: str, choice: str) -> int | None:
    auto = info["roles"].get(role)
    if choice.startswith("Auto"):
        return auto
    if choice == "Select a band":
        return None
    match = re.match(r"Band (\d+):", choice)
    return int(match.group(1)) if match else None


def _pixel_window(ds: Any, bounds: tuple[float, float, float, float], rasterio: Any) -> Any:
    from math import ceil, floor
    from rasterio.windows import Window, from_bounds

    raw = from_bounds(*bounds, transform=ds.transform)
    col0 = max(0, floor(raw.col_off))
    row0 = max(0, floor(raw.row_off))
    col1 = min(ds.width, ceil(raw.col_off + raw.width))
    row1 = min(ds.height, ceil(raw.row_off + raw.height))
    if col1 <= col0 or row1 <= row0:
        raise ValueError("The two satellite rasters do not overlap geographically.")
    return Window(col0, row0, col1 - col0, row1 - row0)


def analyze_pair(
    previous_bytes: bytes,
    current_bytes: bytes,
    previous_red: int,
    previous_nir: int,
    current_red: int,
    current_nir: int,
    threshold: float = 0.10,
    previous_scl: int | None = None,
    current_scl: int | None = None,
    max_side: int = MAX_ANALYSIS_SIDE,
) -> dict[str, Any]:
    """Reproject two multispectral rasters to their common area and compare NDVI."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile
    from rasterio.transform import array_bounds
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_bounds

    if not 0 < threshold < 1:
        raise ValueError("The change threshold must be greater than 0 and less than 1.")
    if not previous_bytes or not current_bytes:
        raise ValueError("Upload both a previous and a current satellite raster.")

    with MemoryFile(previous_bytes) as prev_mem, MemoryFile(current_bytes) as curr_mem:
        with prev_mem.open() as previous, curr_mem.open() as current:
            if previous.crs is None or current.crs is None:
                raise ValueError("Both multispectral rasters need a coordinate reference system (CRS) to align them safely.")
            for label, ds, red_band, nir_band in (
                ("Previous", previous, previous_red, previous_nir),
                ("Current", current, current_red, current_nir),
            ):
                if red_band == nir_band or min(red_band, nir_band) < 1 or max(red_band, nir_band) > ds.count:
                    raise ValueError(f"{label} image needs distinct, valid Red and NIR bands.")

            current_bounds = transform_bounds(current.crs, previous.crs, *current.bounds, densify_pts=21)
            west = max(previous.bounds.left, current_bounds[0])
            south = max(previous.bounds.bottom, current_bounds[1])
            east = min(previous.bounds.right, current_bounds[2])
            north = min(previous.bounds.top, current_bounds[3])
            if east <= west or north <= south:
                raise ValueError("The two rasters do not overlap. Upload images covering the same forest/area.")

            window = _pixel_window(previous, (west, south, east, north), rasterio)
            base_transform = previous.window_transform(window)
            full_width, full_height = int(window.width), int(window.height)
            scale = max(1, int(np.ceil(max(full_width, full_height) / max_side)))
            out_width = max(1, int(np.ceil(full_width / scale)))
            out_height = max(1, int(np.ceil(full_height / scale)))
            from affine import Affine
            target_transform = base_transform * Affine.scale(full_width / out_width, full_height / out_height)
            target_crs = previous.crs

            with WarpedVRT(previous, crs=target_crs, transform=target_transform,
                           width=out_width, height=out_height,
                           resampling=Resampling.bilinear) as prev_vrt, \
                 WarpedVRT(current, crs=target_crs, transform=target_transform,
                           width=out_width, height=out_height,
                           resampling=Resampling.bilinear) as curr_vrt:
                prev_red_data = prev_vrt.read(previous_red, masked=True).astype(np.float32)
                prev_nir_data = prev_vrt.read(previous_nir, masked=True).astype(np.float32)
                curr_red_data = curr_vrt.read(current_red, masked=True).astype(np.float32)
                curr_nir_data = curr_vrt.read(current_nir, masked=True).astype(np.float32)
                prev_valid = ~(np.ma.getmaskarray(prev_red_data) | np.ma.getmaskarray(prev_nir_data))
                curr_valid = ~(np.ma.getmaskarray(curr_red_data) | np.ma.getmaskarray(curr_nir_data))
                prev_red_arr = np.asarray(prev_red_data)
                prev_nir_arr = np.asarray(prev_nir_data)
                curr_red_arr = np.asarray(curr_red_data)
                curr_nir_arr = np.asarray(curr_nir_data)
                prev_den = prev_nir_arr + prev_red_arr
                curr_den = curr_nir_arr + curr_red_arr
                prev_valid &= np.isfinite(prev_red_arr) & np.isfinite(prev_nir_arr) & (prev_den > 0)
                curr_valid &= np.isfinite(curr_red_arr) & np.isfinite(curr_nir_arr) & (curr_den > 0)

                scl_invalid: dict[str, int] = {}
                for source, band, valid, label in (
                    (previous, previous_scl, prev_valid, "previous"),
                    (current, current_scl, curr_valid, "current"),
                ):
                    if band is not None:
                        with WarpedVRT(source, crs=target_crs,
                                       transform=target_transform, width=out_width,
                                       height=out_height, resampling=Resampling.nearest) as scl_vrt:
                            scl_data = scl_vrt.read(band, masked=True)
                        scl_values = np.asarray(scl_data)
                        scl_bad = np.ma.getmaskarray(scl_data) | np.isin(scl_values, [0, 1, 3, 8, 9, 10, 11])
                        valid &= ~scl_bad
                        scl_invalid[label] = int(scl_bad.sum())

                prev_ndvi = np.full((out_height, out_width), np.nan, dtype=np.float32)
                curr_ndvi = np.full((out_height, out_width), np.nan, dtype=np.float32)
                np.divide(prev_nir_arr - prev_red_arr, prev_den, out=prev_ndvi, where=prev_valid)
                np.divide(curr_nir_arr - curr_red_arr, curr_den, out=curr_ndvi, where=curr_valid)
                valid = prev_valid & curr_valid
                delta = np.full((out_height, out_width), np.nan, dtype=np.float32)
                delta[valid] = curr_ndvi[valid] - prev_ndvi[valid]

    valid_count = int(valid.sum())
    if valid_count == 0:
        raise ValueError("No valid overlapping pixels remain after nodata and scene-classification masks.")
    loss = valid & (delta < -threshold)
    improvement = valid & (delta > threshold)
    stable = valid & ~(loss | improvement)
    previous_mean = float(prev_ndvi[valid].mean())
    current_mean = float(curr_ndvi[valid].mean())
    mean_change = float(delta[valid].mean())
    loss_pct = float(loss.sum() / valid_count * 100)
    improvement_pct = float(improvement.sum() / valid_count * 100)
    stable_pct = float(stable.sum() / valid_count * 100)
    if loss_pct > improvement_pct and loss_pct > 0:
        status = "Vegetation Loss Detected"
    elif improvement_pct > loss_pct and improvement_pct > 0:
        status = "Vegetation Improvement Detected"
    else:
        status = "Vegetation Relatively Stable"
    summary = (
        f"{status}: mean NDVI changed by {mean_change:+.4f}; "
        f"{loss_pct:.1f}% of valid overlapping pixels indicate loss, "
        f"{improvement_pct:.1f}% indicate improvement, and {stable_pct:.1f}% are stable."
    )
    return {
        "previous_ndvi": prev_ndvi, "current_ndvi": curr_ndvi,
        "ndvi_change": delta, "valid_mask": valid,
        "summary": {
            "previous_mean_ndvi": previous_mean,
            "current_mean_ndvi": current_mean,
            "mean_ndvi_change": mean_change,
            "vegetation_loss_percentage": loss_pct,
            "vegetation_improvement_percentage": improvement_pct,
            "stable_percentage": stable_pct,
            "vegetation_status": status,
            "analysis_summary": summary,
            "change_threshold": float(threshold),
            "valid_overlap_pixels": valid_count,
            "masked_no_data_cloud_pixels": int(delta.size - valid_count),
            "scene_classification_masked_pixels": scl_invalid,
            "analysis_width": int(out_width), "analysis_height": int(out_height),
            "analysis_crs": target_crs.to_string(),
            "spatial_alignment": "Current raster warped to the previous raster CRS and a common overlapping grid; reported percentages use valid overlapping pixels only.",
            "cloud_mask_note": "Clouds are excluded when a Sentinel-2 SCL scene-classification band is supplied/detected. Otherwise only source nodata/validity masks are excluded.",
            "measurement_provenance": "NDVI=(NIR-Red)/(NIR+Red), calculated from user-selected or metadata-identified raster bands; no RGB-derived NDVI.",
        },
    }


def _ndvi_rgb(ndvi: np.ndarray, valid: np.ndarray) -> np.ndarray:
    stops = np.array([-1.0, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    colors = np.array([
        [92, 45, 20], [150, 72, 28], [194, 112, 46], [221, 186, 74],
        [179, 207, 99], [112, 184, 82], [48, 143, 67], [15, 92, 50],
    ], dtype=np.float32)
    values = np.clip(np.nan_to_num(ndvi, nan=-1.0), -1, 1)
    rgb = np.stack([np.interp(values, stops, colors[:, ch]) for ch in range(3)], axis=-1).astype(np.uint8)
    rgb[~valid] = (32, 42, 38)
    return rgb


def _change_rgb(delta: np.ndarray, valid: np.ndarray, threshold: float) -> np.ndarray:
    rgb = np.zeros((*delta.shape, 3), dtype=np.uint8)
    rgb[:] = (55, 65, 70)
    rgb[valid & (delta < -threshold)] = (220, 65, 65)
    rgb[valid & (delta > threshold)] = (45, 175, 95)
    rgb[valid & (np.abs(delta) <= threshold)] = (228, 188, 68)
    return rgb


def _band_choice(st: Any, info: dict[str, Any], role: str, key: str) -> int | None:
    labels = info["labels"]
    detected = info["roles"].get(role)
    if detected:
        automatic = f"Auto-detected Band {detected}: {labels[detected - 1]}"
        options = [automatic] + [f"Band {i}: {label}" for i, label in enumerate(labels, start=1)]
        chosen = st.selectbox(f"{role.upper()} band", options, key=key)
        if chosen == automatic:
            return detected
    else:
        options = ["Select a band"] + [f"Band {i}: {label}" for i, label in enumerate(labels, start=1)]
        chosen = st.selectbox(f"{role.upper()} band", options, key=key)
    return _selected_band(info, role, chosen)


def _cached_upload_info(st: Any, upload: Any, slot: str) -> dict[str, Any]:
    payload = upload.getvalue()
    fingerprint = hashlib.sha256(payload).hexdigest()
    cache_key = f"sat_compare_{slot}_inspection"
    cached = st.session_state.get(cache_key)
    if cached and cached.get("fingerprint") == fingerprint:
        return cached["info"]
    info = inspect_upload(payload, upload.name)
    st.session_state[cache_key] = {"fingerprint": fingerprint, "info": info}
    return info


def render_satellite_comparison() -> None:
    """Render the upload, alignment, NDVI maps, and evidence integration UI."""
    import streamlit as st

    from wildlife_reporting import save_evidence

    st.header("Satellite / Vegetation Analysis")
    st.caption(
        "Compare two georeferenced multispectral images using their Red and Near-Infrared bands. "
                "RGB photographs are displayed but never treated as NDVI data. Upload a georeferenced, multiband GeoTIFF or JP2 containing both Red and NIR bands for each date."
    )
    previous_col, current_col = st.columns(2)
    with previous_col:
        previous_file = st.file_uploader(
            "Previous Satellite Image", type=["tif", "tiff", "jp2", "png", "jpg", "jpeg"],
            key="sat_compare_previous_upload", help="Earlier multispectral raster; Red and NIR bands are required for NDVI.",
        )
        previous_date = st.date_input("Previous observation date", value=date.today() - timedelta(days=30), key="sat_compare_previous_date")
    with current_col:
        current_file = st.file_uploader(
            "Current Satellite Image", type=["tif", "tiff", "jp2", "png", "jpg", "jpeg"],
            key="sat_compare_current_upload", help="Later multispectral raster of the same area.",
        )
        current_date = st.date_input("Current observation date", value=date.today(), key="sat_compare_current_date")

    previous_info = current_info = None
    comparison_signature = None
    if previous_file is not None or current_file is not None:
        st.subheader("Uploaded image previews")
        preview_cols = st.columns(2)
        for col, upload, label in (
            (preview_cols[0], previous_file, "Previous image"),
            (preview_cols[1], current_file, "Current image"),
        ):
            with col:
                if upload is None:
                    st.info(f"Upload the {label.lower()} to preview it.")
                    continue
                try:
                    info = _cached_upload_info(st, upload, "previous" if label.startswith("Previous") else "current")
                    if label.startswith("Previous"):
                        previous_info = info
                    else:
                        current_info = info
                    st.image(info["preview"], caption=f"{upload.name} · {info['width']} × {info['height']} px", width="stretch")
                    if info["kind"] == "raster":
                        st.caption(f"{info['preview_type']} · {info['count']} bands · CRS: {info['crs'] or 'missing'} · {', '.join(info['labels'])}")
                    else:
                        st.caption("RGB image only · no georeferencing or NIR band")
                except Exception as exc:
                    st.error(f"{label} cannot be previewed: {exc}")

    if previous_info and current_info and previous_info["kind"] == current_info["kind"] == "raster":
        with st.container(border=True):
            st.subheader("Spectral bands and change threshold")
            st.caption("Red/NIR bands are identified from band descriptions and tags when available. Generic band labels require checking the sensor metadata; never select an RGB channel as NIR.")
            p_red_col, p_nir_col, c_red_col, c_nir_col = st.columns(4)
            with p_red_col:
                p_red = _band_choice(st, previous_info, "red", "sat_compare_previous_red")
            with p_nir_col:
                p_nir = _band_choice(st, previous_info, "nir", "sat_compare_previous_nir")
            with c_red_col:
                c_red = _band_choice(st, current_info, "red", "sat_compare_current_red")
            with c_nir_col:
                c_nir = _band_choice(st, current_info, "nir", "sat_compare_current_nir")
            prev_scl = previous_info["roles"].get("scl")
            curr_scl = current_info["roles"].get("scl")
            if prev_scl or curr_scl:
                st.caption("Sentinel-2 scene-classification bands were detected; invalid, cloud-shadow, cloud, cirrus, and snow/ice classes will be masked.")
            if previous_info["rgb_only"] or current_info["rgb_only"]:
                st.warning("At least one upload is an RGB-only raster. It has no NIR band, so true NDVI cannot be calculated.")
            threshold = st.slider(
                "Vegetation-change threshold (absolute ΔNDVI)", min_value=0.01, max_value=0.50,
                value=0.10, step=0.01, key="sat_compare_threshold",
                help="Pixels below −threshold count as loss, above +threshold as improvement, and between them as stable.",
            )
            if previous_date >= current_date:
                st.warning("The current observation date should be later than the previous date for a before/after interpretation.")
            if not previous_info["crs"] or not current_info["crs"]:
                st.warning("Both rasters need a CRS. Assign the correct georeferencing to each raster before comparison.")
            if p_red is None or p_nir is None or c_red is None or c_nir is None:
                st.info("Select Red and NIR bands for both images. Ordinary RGB/JPEG/PNG images do not contain NIR and cannot produce true NDVI.")
            elif not (previous_info["roles"].get("nir") and current_info["roles"].get("nir")):
                st.warning("NIR was not identified from metadata for both images. Verify the selected NIR band against the satellite product metadata before running this analysis.")
            can_run = all((p_red, p_nir, c_red, c_nir, previous_info["crs"], current_info["crs"])) and not (previous_info["rgb_only"] or current_info["rgb_only"])
            if can_run:
                comparison_signature = hashlib.sha256(
                    (previous_info["fingerprint"] + current_info["fingerprint"]
                     + str((p_red, p_nir, c_red, c_nir, threshold,
                            previous_date.isoformat(), current_date.isoformat()))).encode()
                ).hexdigest()
            run = st.button("Run NDVI vegetation comparison", type="primary", disabled=not can_run, key="sat_compare_run")
            if run:
                try:
                    with st.spinner("Aligning overlapping raster pixels and calculating NDVI…"):
                        comparison = analyze_pair(
                            previous_file.getvalue(), current_file.getvalue(),
                            p_red, p_nir, c_red, c_nir, threshold,
                            previous_scl=prev_scl, current_scl=curr_scl,
                        )
                    comparison["summary"].update({
                        "previous_observation_date": previous_date.isoformat(),
                        "current_observation_date": current_date.isoformat(),
                        "previous_filename": previous_file.name,
                        "current_filename": current_file.name,
                        "previous_red_band": p_red, "previous_nir_band": p_nir,
                        "current_red_band": c_red, "current_nir_band": c_nir,
                        "previous_fingerprint": previous_info["fingerprint"],
                        "current_fingerprint": current_info["fingerprint"],
                    })
                    st.session_state["sat_compare_result"] = comparison
                    st.session_state["sat_compare_signature"] = comparison_signature
                    evidence = dict(comparison["summary"])
                    evidence["temporal_features"] = {
                        "date_count": 2, "first_date": previous_date.isoformat(),
                        "last_date": current_date.isoformat(),
                        "mean_ndvi": evidence["current_mean_ndvi"],
                        "latest_ndvi": evidence["current_mean_ndvi"],
                        "overall_ndvi_change": evidence["mean_ndvi_change"],
                        "mean_change": evidence["mean_ndvi_change"],
                    }
                    evidence["temporal_ndvi_series"] = [
                        {"acquisition_date": previous_date.isoformat(), "mean_ndvi": evidence["previous_mean_ndvi"]},
                        {"acquisition_date": current_date.isoformat(), "mean_ndvi": evidence["current_mean_ndvi"]},
                    ]
                    identity = hashlib.sha256(comparison_signature.encode()).hexdigest()
                    save_evidence(
                        st.session_state, "satellite", identity,
                        f"Two-date NDVI comparison: {previous_file.name} → {current_file.name}",
                        measurements=evidence,
                    )
                    st.success("Comparison complete. Structured NDVI evidence has been added to the Reports / fusion pipeline.")
                except Exception as exc:
                    st.error(f"NDVI comparison could not be completed: {exc}")
    elif previous_info and current_info:
        st.warning("True NDVI comparison requires both uploads to be georeferenced multispectral rasters containing Red and NIR bands. RGB-only images are not used to estimate NDVI.")

    result = st.session_state.get("sat_compare_result")
    if result and st.session_state.get("sat_compare_signature") != comparison_signature:
        result = None
        st.info("The uploads or analysis settings changed. Run the comparison again to update these results.")
    if result:
        summary = result["summary"]
        st.subheader(f"Vegetation Status: {summary['vegetation_status']}")
        st.write(summary["analysis_summary"])
        previous_map, current_map, change_map = st.columns(3)
        previous_map.image(
            _ndvi_rgb(result["previous_ndvi"], result["valid_mask"]),
            caption=f"Previous NDVI · mean {summary['previous_mean_ndvi']:.4f}", width="stretch",
        )
        current_map.image(
            _ndvi_rgb(result["current_ndvi"], result["valid_mask"]),
            caption=f"Current NDVI · mean {summary['current_mean_ndvi']:.4f}", width="stretch",
        )
        change_map.image(
            _change_rgb(result["ndvi_change"], result["valid_mask"], summary["change_threshold"]),
            caption=f"NDVI Change · mean {summary['mean_ndvi_change']:+.4f}", width="stretch",
        )
        st.caption("NDVI scale: brown/red = bare or low vegetation · yellow = moderate vegetation · green = dense vegetation. Change map: 🔴 loss · 🟡 stable · 🟢 improvement · dark = masked/no-data.")
        st.caption(f"Change classes use ±{summary['change_threshold']:.2f}. Percentages use valid pixels in the aligned overlapping area; {summary['masked_no_data_cloud_pixels']:,} pixels were masked.")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Vegetation Loss", f"{summary['vegetation_loss_percentage']:.1f}%")
        metric_cols[1].metric("Vegetation Improvement", f"{summary['vegetation_improvement_percentage']:.1f}%")
        metric_cols[2].metric("Stable Vegetation", f"{summary['stable_percentage']:.1f}%")
        means = st.columns(3)
        means[0].metric("Mean Previous NDVI", f"{summary['previous_mean_ndvi']:.4f}")
        means[1].metric("Mean Current NDVI", f"{summary['current_mean_ndvi']:.4f}")
        means[2].metric("Mean NDVI Change", f"{summary['mean_ndvi_change']:+.4f}")
        with st.expander("Spatial validation and processing details"):
            st.write(summary["spatial_alignment"])
            st.write(summary["cloud_mask_note"])
            st.json({key: value for key, value in summary.items() if key not in {"analysis_summary"}})
        st.download_button(
            "Download structured NDVI result (JSON)",
            data=json.dumps(summary, indent=2),
            file_name="satellite_vegetation_comparison.json",
            mime="application/json",
            key="sat_compare_download",
        )
