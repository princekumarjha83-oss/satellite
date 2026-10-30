"""satellite_ndvi.py — Satellite / NDVI Monitoring page for EcoMRV-Agent.

Discovers real Sentinel-2 Level-2A SAFE archives (folders or ZIP files),
calculates NDVI from B04 (red) and B08 (NIR) JP2 bands, shows temporal
vegetation change, and feeds calibrated satellite evidence into the
existing multi-modal fusion pipeline.

Nothing in this module fabricates NDVI.  All values are calculated from
real Sentinel-2 B04 / B08 bands.  NDVI change is NOT a deforestation
determination.  Human review is required before drawing environmental
conclusions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

from ecomrv_config import NDVI_CHANGE_THRESHOLDS
from ecosystem_pipeline import satellite_features
from satellite_temporal import (
    _vsi_zip_path,
    analyze_safe_archives,
    discover_safe_zips,
    get_default_safe_search_locations,
    inspect_safe_archive,
)
from ui_state import saved_control
from wildlife_reporting import save_evidence

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_THUMBNAIL_SIZE = 300
_OUTPUT_DIR = Path("data") / "satellite_outputs"

# Locations searched for SAFE archives (parent dir, data/satellite, cwd)
_SEARCH_LOCATIONS = get_default_safe_search_locations()

# RdYlGn colorscale: red = sparse / bare, green = dense vegetation
_NDVI_COLORSCALE = [
    [0.00, "#8b0000"], [0.15, "#d73027"], [0.30, "#fc8d59"],
    [0.43, "#fee090"], [0.55, "#d9ef8b"], [0.70, "#91cf60"],
    [1.00, "#1a9850"],
]
# Diverging colorscale for NDVI change: red = decline, white = stable, green = increase
_CHANGE_COLORSCALE = [
    [0.00, "#d73027"], [0.35, "#fc8d59"], [0.48, "#fee090"],
    [0.50, "#ffffbf"], [0.52, "#d9ef8b"], [0.65, "#91cf60"],
    [1.00, "#1a9850"],
]

_PLOT_BG = "#050d08"
_PLOT_GRID = "#1a3d26"
_PLOT_FG = "#e2f5e9"
_PLOT_TICK = "#a7f3d0"
_PLOT_LINE = "#22c55e"
_LEGEND_BG = "rgba(10,26,15,0.85)"


# ─────────────────────────────────────────────────────────────────────────────
# Thumbnail helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_geotiff_thumbnail(tif_path: Path, size: int) -> np.ndarray | None:
    """Decimated read from a GeoTIFF; returns a 2-D float32 array or None."""
    try:
        import rasterio
    except ImportError:
        return None
    try:
        with rasterio.open(str(tif_path)) as ds:
            h, w = ds.height, ds.width
            out_h = max(1, int(size * h / w))
            arr = ds.read(1, out_shape=(1, out_h, size)).astype(np.float32)
        arr[arr == -9999.0] = np.nan
        return arr[0]
    except Exception:
        return None


def _ndvi_thumbnail(scene: dict[str, Any], size: int = _THUMBNAIL_SIZE) -> np.ndarray | None:
    """
    Return a small NDVI array for display.

    Read a decimated thumbnail directly from the calibrated source bands.
    Saved GeoTIFFs can be block-sampled for speed and therefore contain nodata
    gaps; they are not suitable as display images.

    All pixel values are real — nothing is fabricated.
    """
    # Read only a small display grid from the source JP2 bands. This avoids
    # showing incomplete GeoTIFFs when analysis used block sampling.
    try:
        import rasterio
    except ImportError:
        return None
    try:
        is_zip = scene.get("is_zip", False)
        red_p = _vsi_zip_path(scene["source"], scene["band_members"]["B04_red"], is_zip=is_zip)
        nir_p = _vsi_zip_path(scene["source"], scene["band_members"]["B08_nir"], is_zip=is_zip)
        with rasterio.open(red_p) as red_ds, rasterio.open(nir_p) as nir_ds:
            h, w = red_ds.height, red_ds.width
            # Read a small, aspect-correct thumbnail and preserve nodata masks.
            out_h = max(1, round(size * h / w))
            out_shape = (out_h, size)
            r = red_ds.read(1, out_shape=out_shape, masked=True).astype(np.float32)
            n = nir_ds.read(1, out_shape=out_shape, masked=True).astype(np.float32)
        valid = ~(np.ma.getmaskarray(r) | np.ma.getmaskarray(n))
        rv, nv = np.asarray(r), np.asarray(n)
        denom = nv + rv
        ndvi = np.full(out_shape, np.nan, dtype=np.float32)
        np.divide(nv - rv, denom, out=ndvi, where=valid & (denom != 0))
        return np.clip(ndvi, -1.0, 1.0)
    except Exception:
        return None


def _change_thumbnail(older: dict[str, Any], newer: dict[str, Any],
                      size: int = _THUMBNAIL_SIZE) -> np.ndarray | None:
    """
    Return a small NDVI-change array for a spatially compatible scene pair.
    Returns None if the scenes cannot be pixel-compared.
    """
    if (older.get("tile_id") != newer.get("tile_id") or
            older.get("crs") != newer.get("crs") or
            older.get("raster_dimensions") != newer.get("raster_dimensions")):
        return None

    # Compute the display layer from source thumbnails. A saved change GeoTIFF
    # may be from an older sampled run and contain nodata gaps.
    old_arr = _ndvi_thumbnail(older, size)
    new_arr = _ndvi_thumbnail(newer, size)
    if old_arr is None or new_arr is None or old_arr.shape != new_arr.shape:
        return None
    return np.clip(new_arr - old_arr, -1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Plotly chart builders
# ─────────────────────────────────────────────────────────────────────────────

def _timeseries_chart(series: list[dict]) -> Any:
    """Line + error-bar chart of mean NDVI per acquisition date/tile."""
    import plotly.graph_objects as go

    labels = [f"{s['acquisition_date']}<br>Tile {s.get('tile_id', '?')}" for s in series]
    means   = [round(float(s["mean_ndvi"]), 4)                          for s in series]
    medians = [round(float(s.get("median_ndvi", s["mean_ndvi"])), 4)    for s in series]
    stds    = [round(float(s.get("std_ndvi", 0.0)), 4)                  for s in series]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=means, mode="lines+markers", name="Mean NDVI",
        line=dict(color=_PLOT_LINE, width=3),
        marker=dict(size=11, color=_PLOT_LINE, line=dict(color="#ffffff", width=1)),
        error_y=dict(type="data", array=stds, color="rgba(34,197,94,0.35)", thickness=2),
        hovertemplate="<b>%{x}</b><br>Mean NDVI: %{y:.4f} ± %{error_y.array:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=medians, mode="markers", name="Median NDVI",
        marker=dict(size=9, color="#a3e635", symbol="diamond"),
        hovertemplate="<b>%{x}</b><br>Median NDVI: %{y:.4f}<extra></extra>",
    ))
    for ndvi_val, label, color in [
        (0.3, "Vegetation threshold (0.3)", "#d9ef8b"),
        (0.6, "Dense vegetation (0.6)",     "#1a9850"),
    ]:
        fig.add_hline(y=ndvi_val, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_font_color=color,
                      annotation_position="top right")
    fig.update_layout(
        height=380, paper_bgcolor=_PLOT_BG, plot_bgcolor="#0a1a0f",
        font=dict(color=_PLOT_FG),
        xaxis=dict(gridcolor=_PLOT_GRID, showgrid=True,
                   title="Acquisition Date / Tile", tickfont=dict(color=_PLOT_TICK)),
        yaxis=dict(gridcolor=_PLOT_GRID, showgrid=True,
                   title="NDVI", range=[-0.1, 1.0], tickfont=dict(color=_PLOT_TICK)),
        legend=dict(bgcolor=_LEGEND_BG, bordercolor=_PLOT_LINE,
                    borderwidth=1, font=dict(color=_PLOT_FG)),
        margin=dict(l=55, r=20, t=30, b=55),
    )
    return fig


def _ndvi_heatmap(arr: np.ndarray, title: str, subtitle: str) -> Any:
    """Plotly heatmap of an NDVI thumbnail array using RdYlGn colorscale."""
    import plotly.graph_objects as go

    vmin, vmax = -0.2, 0.8
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    display = np.where(np.isnan(arr), None, norm)

    fig = go.Figure(go.Heatmap(
        z=display, colorscale=_NDVI_COLORSCALE,
        zmin=0.0, zmax=1.0, showscale=True,
        colorbar=dict(
            tickvals=[0.0, 0.25, 0.5, 0.75, 1.0],
            ticktext=[f"{vmin:.1f}",
                      f"{vmin + (vmax - vmin) * 0.25:.2f}",
                      f"{(vmin + vmax) / 2:.2f}",
                      f"{vmin + (vmax - vmin) * 0.75:.2f}",
                      f"{vmax:.1f}"],
            title=dict(text="NDVI", font=dict(color=_PLOT_FG)),
            tickfont=dict(color=_PLOT_FG),
        ),
        hovertemplate="NDVI ≈ %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>{subtitle}</sub>",
                   font=dict(color=_PLOT_LINE, size=13)),
        height=340, paper_bgcolor=_PLOT_BG, plot_bgcolor="#0a1a0f",
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, scaleanchor="x"),
        margin=dict(l=10, r=75, t=60, b=10),
    )
    return fig


def _ndvi_rgb_image(arr: np.ndarray) -> np.ndarray:
    """Convert an NDVI thumbnail to a displayable RGB image."""
    stops = np.array([0.00, 0.15, 0.30, 0.43, 0.55, 0.70, 1.00])
    colors = np.array([
        [139, 0, 0], [215, 48, 39], [252, 141, 89], [254, 224, 144],
        [217, 239, 139], [145, 207, 96], [26, 152, 80],
    ], dtype=np.float32)
    valid = np.isfinite(arr)
    scaled = np.clip((np.nan_to_num(arr.astype(np.float32), nan=-0.2) + 0.2) / 1.0, 0.0, 1.0)
    rgb = np.empty((*arr.shape, 3), dtype=np.uint8)
    for channel in range(3):
        rgb[..., channel] = np.interp(scaled, stops, colors[:, channel]).astype(np.uint8)
    rgb[~valid] = (10, 26, 15)
    return rgb


def _change_rgb_image(arr: np.ndarray) -> np.ndarray:
    """Convert an NDVI-change thumbnail to a red-yellow-green RGB image."""
    stops = np.array([0.00, 0.35, 0.48, 0.50, 0.52, 0.65, 1.00])
    colors = np.array([
        [215, 48, 39], [252, 141, 89], [254, 224, 144], [255, 255, 191],
        [217, 239, 139], [145, 207, 96], [26, 152, 80],
    ], dtype=np.float32)
    valid = np.isfinite(arr)
    values = np.nan_to_num(arr.astype(np.float32), nan=0.0)
    valid_values = values[valid]
    abs_max = float(np.percentile(np.abs(valid_values), 95)) if valid_values.size else 0.05
    abs_max = max(abs_max, 0.02)
    scaled = np.clip((values + abs_max) / (2.0 * abs_max), 0.0, 1.0)
    rgb = np.empty((*arr.shape, 3), dtype=np.uint8)
    for channel in range(3):
        rgb[..., channel] = np.interp(scaled, stops, colors[:, channel]).astype(np.uint8)
    rgb[~valid] = (10, 26, 15)
    return rgb


def _change_heatmap(arr: np.ndarray, title: str, subtitle: str) -> Any:
    """Plotly heatmap of an NDVI change array (diverging, symmetric around 0)."""
    import plotly.graph_objects as go

    valid = arr[~np.isnan(arr)]
    abs_max = float(np.percentile(np.abs(valid), 95)) if valid.size else 0.05
    abs_max = max(abs_max, 0.02)

    fig = go.Figure(go.Heatmap(
        z=np.where(np.isnan(arr), None, arr),
        colorscale=_CHANGE_COLORSCALE,
        zmid=0.0, zmin=-abs_max, zmax=abs_max,
        showscale=True,
        colorbar=dict(
            tickvals=[-abs_max, -abs_max / 2, 0, abs_max / 2, abs_max],
            ticktext=[f"{-abs_max:.3f}", f"{-abs_max / 2:.3f}", "0",
                      f"{abs_max / 2:.3f}", f"{abs_max:.3f}"],
            title=dict(text="ΔNDVI", font=dict(color=_PLOT_FG)),
            tickfont=dict(color=_PLOT_FG),
        ),
        hovertemplate="ΔNDVI: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>{subtitle}</sub>",
                   font=dict(color=_PLOT_LINE, size=13)),
        height=340, paper_bgcolor=_PLOT_BG, plot_bgcolor="#0a1a0f",
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, scaleanchor="x"),
        margin=dict(l=10, r=75, t=60, b=10),
    )
    return fig


def _vegetation_bar(scenes: list[dict]) -> Any:
    """Stacked bar chart of vegetation coverage classes per scene."""
    import plotly.graph_objects as go

    labels = [f"{s['acquisition_date']}<br>Tile {s.get('tile_id', '?')}" for s in scenes]
    low    = [float(s.get("low_ndvi_pct", 0.0))            for s in scenes]
    high   = [float(s.get("high_ndvi_pct", 0.0))           for s in scenes]
    veg    = [float(s.get("vegetation_coverage_pct", 0.0)) for s in scenes]
    medium = [max(0.0, v - h) for v, h in zip(veg, high)]
    other  = [max(0.0, 100.0 - lo - med - hi)
              for lo, med, hi in zip(low, medium, high)]

    fig = go.Figure()
    for name, values, color in [
        ("Other / NoData",                other,  "#4a5568"),
        ("Sparse / Bare (NDVI < 0.2)",    low,    "#d73027"),
        ("Moderate vegetation (0.2–0.6)", medium, "#fee090"),
        ("Dense vegetation (NDVI ≥ 0.6)", high,   "#1a9850"),
    ]:
        fig.add_trace(go.Bar(
            name=name, x=labels, y=values, marker_color=color,
            hovertemplate=f"{name}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", height=340, paper_bgcolor=_PLOT_BG, plot_bgcolor="#0a1a0f",
        font=dict(color=_PLOT_FG),
        xaxis=dict(gridcolor=_PLOT_GRID, title="Acquisition Date / Tile",
                   tickfont=dict(color=_PLOT_TICK)),
        yaxis=dict(gridcolor=_PLOT_GRID, title="% of valid pixels",
                   range=[0, 105], tickfont=dict(color=_PLOT_TICK)),
        legend=dict(bgcolor=_LEGEND_BG, bordercolor=_PLOT_LINE,
                    borderwidth=1, font=dict(color=_PLOT_FG)),
        margin=dict(l=55, r=20, t=20, b=55),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# ██  PAGE START  ████████████████████████████████████████████████████████████
# ═════════════════════════════════════════════════════════════════════════════

st.header("🛰️ Satellite / NDVI Monitoring")
st.caption(
    "Multi-temporal Sentinel-2 Level-2A vegetation monitoring. "
    "NDVI is calculated **only** from real B04 (red) and B08 (NIR) JP2 bands — "
    "never from display imagery. Results feed directly into the multi-modal "
    "fusion pipeline (fusion score, agreement score, anomaly score, MEHS)."
)

# ─── 1. Scene Discovery ───────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("📁 Discovered Sentinel-2 Archives")
    paths = discover_safe_zips(_SEARCH_LOCATIONS)

    if not paths:
        st.warning(
            "No Sentinel-2 Level-2A SAFE archives or ZIP files found in: "
            + ", ".join(str(loc) for loc in _SEARCH_LOCATIONS)
        )
        st.info(
            "Place `.SAFE` folders or `_SAFE.zip` archives in the project parent "
            "directory (`../`), `data/satellite/`, or the current working directory, "
            "then reload."
        )
        st.stop()

    # Inspect metadata for all archives (zero pixel reads at this stage)
    scene_metas: list[dict] = []
    for path in paths:
        try:
            scene_metas.append(inspect_safe_archive(path))
        except Exception as exc:
            st.error(f"Cannot read **{path.name}**: {exc}")

    if not scene_metas:
        st.error("No archives could be inspected. Check file integrity.")
        st.stop()

    # --- Metadata table ---
    meta_rows = [{
        "Archive":    m["original_filename"],
        "Date":       m["acquisition_date"],
        "Tile":       m.get("tile_id") or "—",
        "CRS":        m.get("crs") or "—",
        "Cloud %":    f"{m['cloud_cover_pct']:.1f}" if m.get("cloud_cover_pct") is not None else "—",
        "Resolution": f"{m.get('resolution_m', 10)} m",
        "Format":     "ZIP" if m.get("is_zip") else "Folder",
        "B04 ✓":      "✅" if m["band_members"].get("B04_red") else "❌",
        "B08 ✓":      "✅" if m["band_members"].get("B08_nir") else "❌",
    } for m in scene_metas]
    st.dataframe(meta_rows, hide_index=True, use_container_width=True)

    tiles      = sorted({m["tile_id"] for m in scene_metas if m.get("tile_id")})
    acq_dates  = sorted({m["acquisition_date"] for m in scene_metas})
    st.caption(
        f"**{len(scene_metas)} archive(s)** · **{len(tiles)} tile(s)** "
        f"({', '.join(tiles)}) · **{len(acq_dates)} unique date(s)** "
        f"({', '.join(acq_dates)}). Metadata read — no pixels loaded."
    )
    if len(tiles) > 1:
        st.info(
            "ℹ️ **Multiple tiles detected.** Only scenes sharing the same tile, CRS, "
            "and spatial grid will be compared at pixel level. Different tiles are "
            "processed independently and are not merged or reprojected."
        )

# ─── 2. Analysis Settings ─────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("⚙️ Analysis Settings")
    col_l, col_r = st.columns(2)
    with col_l:
        sample_blocks = saved_control(
            st.selectbox, "ndvi_sample_blocks",
            "Processing speed (block sampling rate)",
            [1, 2, 4, 8], index=3,
            help=(
                "1 = full resolution (slow, most complete); "
                "4 = recommended balance; 8 = fastest (less complete). "
                "Sampled pixels are real; no values are fabricated."
            ),
        )
    with col_r:
        save_rasters = saved_control(
            st.checkbox, "ndvi_save_rasters",
            "Save NDVI / change GeoTIFFs to disk",
            False,
            help=(
                "Writes compressed float32 GeoTIFF files to data/satellite_outputs/ "
                "for use in external GIS software."
            ),
        )
    # Sampled runs are intended for quick interactive review. Write complete
    # GeoTIFFs only for full-resolution processing to avoid sparse rasters.
    output_dir = _OUTPUT_DIR if save_rasters and sample_blocks == 1 else None
    if save_rasters and sample_blocks != 1:
        st.info("GeoTIFF export requires processing speed **1 (full resolution)**. Current run uses sampled values for a faster preview.")
    st.caption(
        f"Sentinel-2 B04 + B08 at 10 m resolution. "
        f"Sampling every **{sample_blocks}** block(s). "
        "Windowed reads keep memory use bounded; no full rasters are loaded at once."
    )

# ─── 3. Analyse / Clear ───────────────────────────────────────────────────
fingerprint = hashlib.sha256(
    json.dumps(
        [m["sha256"] for m in scene_metas] + [sample_blocks],
        sort_keys=True,
    ).encode()
).hexdigest()

col_btn, col_clr = st.columns([5, 1])
with col_btn:
    run_analysis = st.button(
        f"🔬  Analyse {len(scene_metas)} archive(s) — calculate NDVI from B04 / B08",
        type="primary", use_container_width=True,
        help="Reads real JP2 bands; no synthetic NDVI values are produced.",
    )
with col_clr:
    if st.button("🗑️ Clear", use_container_width=True):
        for key in ("satellite_ndvi_result", "satellite_ndvi_fp",
                    "satellite_ndvi_thumbs", "satellite_ndvi_change_thumbs"):
            st.session_state.pop(key, None)
        st.rerun()

if run_analysis:
    for key in ("satellite_ndvi_result", "satellite_ndvi_thumbs",
                "satellite_ndvi_change_thumbs"):
        st.session_state.pop(key, None)
    try:
        with st.spinner(
            f"Reading Sentinel-2 B04 / B08 JP2 bands and calculating NDVI "
            f"(sampling every {sample_blocks} block(s)) …"
        ):
            result = analyze_safe_archives(
                paths, output_dir=output_dir, sample_blocks=sample_blocks,
            )
        st.session_state["satellite_ndvi_result"] = result
        st.session_state["satellite_ndvi_fp"]     = fingerprint
        # ── Register evidence in the shared pipeline ───────────────────────
        identity = hashlib.sha256(
            json.dumps(
                [s["sha256"] for s in result["scenes"]], sort_keys=True
            ).encode()
        ).hexdigest()
        save_evidence(
            st.session_state, "satellite", identity,
            f"{result['date_count']} Sentinel-2 L2A observation(s)",
            measurements=result,
        )
        st.success(
            f"✅ NDVI calculated for **{result['date_count']} scene(s)**. "
            "Satellite evidence registered in the pipeline — "
            "navigate to **Reports** to generate the combined multi-modal report."
        )
    except RuntimeError as exc:
        st.error(str(exc))
        st.info("Install rasterio: `pip install rasterio`")
        st.stop()
    except Exception as exc:
        st.error(f"NDVI analysis could not complete: {exc}")
        st.stop()

# ─── Guard: nothing to show yet ───────────────────────────────────────────
result = st.session_state.get("satellite_ndvi_result")
if result is None:
    st.info(
        "Click **Analyse** above to calculate NDVI from the discovered archives. "
        "Temporal charts, vegetation maps, and change detection will appear here."
    )
    st.stop()

# Unpack convenience references
scenes:  list[dict] = result["scenes"]
series:  list[dict] = result["temporal_ndvi_series"]
changes: list[dict] = result.get("changes", [])
anomaly: dict       = result.get("temporal_anomaly", {})
tf:      dict       = result.get("temporal_features", {})

# ─────────────────────────────────────────────────────────────────────────────
# ██  RESULTS  ████████████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

# ─── 4a. NDVI Temporal Series ─────────────────────────────────────────────
with st.container(border=True):
    st.subheader("📈 NDVI Temporal Series")
    st.plotly_chart(_timeseries_chart(series), use_container_width=True)

    stat_rows = [{
        "Date":           s["acquisition_date"],
        "Tile":           s.get("tile_id") or "—",
        "Mean NDVI":      f"{s['mean_ndvi']:.4f}",
        "Median NDVI":    f"{s.get('median_ndvi', float('nan')):.4f}",
        "Std Dev":        f"{s.get('std_ndvi', float('nan')):.4f}",
        "Min NDVI":       f"{s['min_ndvi']:.4f}"         if "min_ndvi"       in s else "—",
        "Max NDVI":       f"{s['max_ndvi']:.4f}"         if "max_ndvi"       in s else "—",
        "Dense veg %":    f"{s['high_ndvi_pct']:.1f}%"   if "high_ndvi_pct"  in s else "—",
        "Sparse/bare %":  f"{s['low_ndvi_pct']:.1f}%"    if "low_ndvi_pct"   in s else "—",
        "Valid pixels":   f"{s.get('valid_pixels', 0):,}",
    } for s in scenes]
    st.dataframe(stat_rows, hide_index=True, use_container_width=True)
    st.caption("All statistics computed from sampled B04 / B08 pixel data. No values invented.")

# ─── 4b. Vegetation Coverage ──────────────────────────────────────────────
with st.container(border=True):
    st.subheader("🌿 Vegetation Coverage by Scene")
    st.plotly_chart(_vegetation_bar(scenes), use_container_width=True)
    st.caption(
        f"Project thresholds (not validated deforestation thresholds) — "
        f"sparse/bare: NDVI < 0.2 · moderate: 0.2–0.6 · dense: NDVI ≥ 0.6. "
        "NDVI reflects all land cover in the tile, not exclusively forest canopy."
    )

# ─── 4c. NDVI Maps (thumbnails) ───────────────────────────────────────────
with st.container(border=True):
    st.subheader("🗺️ NDVI Maps")
    st.caption(
        "Decimated from real B04 / B08 bands (pre-computed GeoTIFF preferred; "
        "JP2 decimated read as fallback). Display resolution ≈ 300 × 300 px. "
        "No pixel values are interpolated or fabricated."
    )

    if "satellite_ndvi_thumbs" not in st.session_state:
        thumbs: dict[str, np.ndarray | None] = {}
        for sc in scenes:
            with st.spinner(
                f"Generating NDVI map — {sc['acquisition_date']} "
                f"tile {sc.get('tile_id', '?')} …"
            ):
                thumbs[sc["original_filename"]] = _ndvi_thumbnail(sc)
        st.session_state["satellite_ndvi_thumbs"] = thumbs
    thumbs = st.session_state["satellite_ndvi_thumbs"]

    cols_per_row = min(len(scenes), 3)
    for batch_start in range(0, len(scenes), cols_per_row):
        batch = scenes[batch_start : batch_start + cols_per_row]
        cols  = st.columns(len(batch))
        for col, sc in zip(cols, batch):
            arr = thumbs.get(sc["original_filename"])
            with col:
                if arr is not None and not np.all(np.isnan(arr)):
                    st.image(
                        _ndvi_rgb_image(arr),
                        caption=(
                            f"Tile {sc.get('tile_id', '?')} · {sc['acquisition_date']} "
                            "· NDVI (red = sparse, green = dense vegetation)"
                        ),
                        width="stretch",
                    )
                    st.caption(
                        f"Mean NDVI: **{sc['mean_ndvi']:.4f}** · "
                        f"Dense veg: **{sc.get('high_ndvi_pct', 0.0):.1f}%** · "
                        f"Valid px: {sc.get('valid_pixels', 0):,}"
                    )
                else:
                    st.warning(
                        f"NDVI map unavailable — {sc['acquisition_date']} / "
                        f"tile {sc.get('tile_id', '?')}. "
                        "Ensure rasterio is installed and JP2 files are accessible."
                    )

# ─── 4d. NDVI Change Detection ────────────────────────────────────────────
with st.container(border=True):
    st.subheader("🔄 NDVI Change Detection")
    if not changes:
        st.info("At least two scenes are required for change detection.")
    else:
        change_rows = []
        for ch in changes:
            period = f"{ch['from_date']} → {ch['to_date']}"
            ctype  = ch.get("label", "").replace("_", " ").title()
            tile   = ch.get("from_tile") or "—"
            if ch.get("status") == "available":
                mean_ch  = float(ch.get("mean_change") or 0.0)
                valid_px = max(int(ch.get("valid_pixels") or 1), 1)
                dec_pct  = ch.get("vegetation_decrease_pixels", 0) / valid_px * 100
                inc_pct  = ch.get("vegetation_increase_pixels", 0) / valid_px * 100
                stb_pct  = ch.get("stable_pixels",              0) / valid_px * 100
                if mean_ch <= NDVI_CHANGE_THRESHOLDS["decrease"]:
                    signal = "🔴 Vegetation decline signal"
                elif mean_ch >= NDVI_CHANGE_THRESHOLDS["increase"]:
                    signal = "🟢 Vegetation increase signal"
                else:
                    signal = "🟡 Relatively stable"
                change_rows.append({
                    "Period":            period,
                    "Type":              ctype,
                    "Tile":              tile,
                    "Mean ΔNDVI":        f"{mean_ch:+.4f}",
                    "Decline signal %":  f"{dec_pct:.1f}%",
                    "Stable %":          f"{stb_pct:.1f}%",
                    "Increase signal %": f"{inc_pct:.1f}%",
                    "Vegetation signal": signal,
                    "Pixel status":      "✅ Pixel comparison performed",
                })
            else:
                change_rows.append({
                    "Period":            period,
                    "Type":              ctype,
                    "Tile":              tile,
                    "Mean ΔNDVI":        "—",
                    "Decline signal %":  "—",
                    "Stable %":          "—",
                    "Increase signal %": "—",
                    "Vegetation signal": "—",
                    "Pixel status":      f"⚠️ {ch.get('reason', 'Not compared')}",
                })
        st.dataframe(change_rows, hide_index=True, use_container_width=True)
        st.caption(
            f"Project thresholds — decline: ΔNDVI ≤ {NDVI_CHANGE_THRESHOLDS['decrease']} · "
            f"stable: |ΔNDVI| ≤ {NDVI_CHANGE_THRESHOLDS['stable']} · "
            f"increase: ΔNDVI ≥ {NDVI_CHANGE_THRESHOLDS['increase']}. "
            "**NDVI change is not a deforestation or illegal-activity determination. "
            "Human expert review is required before drawing environmental conclusions.**"
        )

# ─── 4e. NDVI Change Maps ─────────────────────────────────────────────────
available_changes = [ch for ch in changes if ch.get("status") == "available"]
if available_changes:
    with st.container(border=True):
        st.subheader("🗺️ NDVI Change Maps")
        st.caption(
            "🔴 Red = vegetation decline signal · 🟡 Yellow = stable · 🟢 Green = increase. "
            "This is a vegetation-change signal, **not** a deforestation determination."
        )

        if "satellite_ndvi_change_thumbs" not in st.session_state:
            ct: dict[str, np.ndarray | None] = {}
            for ch in available_changes:
                older_sc = next(
                    (s for s in scenes
                     if s["acquisition_date"] == ch["from_date"]
                     and s.get("tile_id") == ch.get("from_tile")),
                    None,
                )
                newer_sc = next(
                    (s for s in scenes
                     if s["acquisition_date"] == ch["to_date"]
                     and s.get("tile_id") == ch.get("to_tile")),
                    None,
                )
                ckey = f"{ch['from_date']}_{ch['to_date']}_{ch.get('from_tile', '')}"
                if older_sc and newer_sc:
                    with st.spinner(
                        f"Generating change map {ch['from_date']} → {ch['to_date']} …"
                    ):
                        ct[ckey] = _change_thumbnail(older_sc, newer_sc)
                else:
                    ct[ckey] = None
            st.session_state["satellite_ndvi_change_thumbs"] = ct
        ct = st.session_state["satellite_ndvi_change_thumbs"]

        cols_per_row = min(len(available_changes), 2)
        for batch_start in range(0, len(available_changes), cols_per_row):
            batch = available_changes[batch_start : batch_start + cols_per_row]
            cols  = st.columns(len(batch))
            for col, ch in zip(cols, batch):
                ckey = f"{ch['from_date']}_{ch['to_date']}_{ch.get('from_tile', '')}"
                arr  = ct.get(ckey)
                with col:
                    if arr is not None and not np.all(np.isnan(arr)):
                        mean_ch = float(ch.get("mean_change") or 0.0)
                        st.image(
                            _change_rgb_image(arr),
                            caption=(
                                f"Tile {ch.get('from_tile', '?')} · "
                                f"{ch['from_date']} → {ch['to_date']} · "
                                "red = decline, yellow = stable, green = increase"
                            ),
                            width="stretch",
                        )
                        st.caption(
                            f"Mean ΔNDVI: **{mean_ch:+.4f}** — "
                            f"{ch.get('interpretation', 'Vegetation-change signal from aligned NDVI.')}"
                        )
                    else:
                        st.warning(
                            f"Change map unavailable for "
                            f"{ch['from_date']} → {ch['to_date']}."
                        )

# ─── 4f. Temporal Anomaly ─────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("⚡ Temporal Anomaly Assessment")
    if anomaly.get("anomaly_score") is not None:
        a_score = float(anomaly["anomaly_score"])
        is_anom = bool(anomaly.get("is_anomaly", False))
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Anomaly Score",     f"{a_score:.1f} / 100")
        col2.metric("Anomaly Detected",  "Yes ⚠️" if is_anom else "No ✅")
        col3.metric("Current NDVI",      f"{anomaly.get('current_value', 0.0):.4f}")
        col4.metric("Historical NDVI (baseline)",
                    f"{anomaly.get('historical_baseline', 0.0):.4f}"
                    if anomaly.get("historical_baseline") is not None else "—")
        st.caption(anomaly.get("reason", ""))
        if anomaly.get("latest_change") is not None:
            st.caption(
                f"Latest NDVI change used in anomaly context: "
                f"**{anomaly['latest_change']:+.4f}** · "
                f"Temporal variability (std): **{anomaly.get('temporal_variability', 0.0):.4f}**"
            )
    else:
        reason = anomaly.get(
            "reason",
            "A same-tile series with at least three distinct acquisition dates is required.",
        )
        st.info(f"ℹ️ {reason}")
    st.caption(
        "Anomaly indicates statistical deviation from the preceding NDVI series. "
        "It does **not** identify illegal activity, deforestation, or environmental damage. "
        "Human expert review is required."
    )

# ─── 4g. Spatial Validation ───────────────────────────────────────────────
with st.expander("📐 Spatial Validation Details"):
    if len(scenes) < 2:
        st.info("At least two scenes are required for spatial comparison.")
    else:
        for ch in changes:
            ok = ch.get("status") == "available"
            icon = "✅" if ok else "⚠️"
            tile_str = ch.get("from_tile") or "?"
            st.markdown(
                f"{icon} **{ch['from_date']} → {ch['to_date']}** "
                f"(tile {tile_str})"
            )
            if ok:
                st.caption(
                    "Same tile, CRS, spatial transform, and raster dimensions confirmed. "
                    "Pixel-level NDVI comparison was performed."
                )
            else:
                st.caption(f"Not compared: {ch.get('reason', 'scenes not compatible')}")
        st.caption(
            "Spatial validation checks: tile ID · CRS · raster dimensions · "
            "spatial transform · geographic overlap. "
            "No reprojection or resampling is performed automatically."
        )

# ─────────────────────────────────────────────────────────────────────────────
# ██  PIPELINE CONTRIBUTION  ██████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

with st.container(border=True):
    st.subheader("🔗 Satellite Features & Multi-Modal Pipeline")
    sat_feat = satellite_features(result)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Mean NDVI (series)",
        f"{tf['mean_ndvi']:.4f}" if tf.get("mean_ndvi") is not None else "—",
    )
    col2.metric(
        "Latest NDVI",
        f"{tf['latest_ndvi']:.4f}" if tf.get("latest_ndvi") is not None else "—",
    )
    col3.metric(
        "Overall ΔNDVI",
        f"{tf['overall_ndvi_change']:+.4f}" if tf.get("overall_ndvi_change") is not None else "—",
        help="Change from the first to the latest scene. Uses pixel-aligned NDVI when available; otherwise uses the difference between scene mean NDVI values for a single tile.",
    )
    col4.metric(
        "Normalised fusion score",
        f"{sat_feat['normalized_score']:.1f} / 100"
        if sat_feat.get("normalized_score") is not None else "Insufficient",
    )
    if tf.get("overall_change_method") == "difference between first and latest scene mean NDVI":
        st.caption("Overall ΔNDVI uses the first-to-latest scene mean difference because their raster grids could not be pixel-aligned. See Spatial Validation Details for the pixel-comparison status.")

    if sat_feat.get("normalized_score") is not None:
        st.success(
            f"✅ **Satellite evidence available** — normalised score: "
            f"{sat_feat['normalized_score']:.1f} / 100 "
            f"(derived from NDVI {tf.get('latest_ndvi') or tf.get('mean_ndvi'):.4f}). "
            "This contributes to the multi-modal fusion score in the Reports page."
        )
    else:
        st.warning(
            f"⚠️ **Satellite evidence insufficient for fusion**: "
            f"{sat_feat.get('reason', 'no calibrated NDVI available')}. "
            "A same-tile series with at least one valid NDVI is required for fusion contribution."
        )

    col_feat, col_tf = st.columns(2)
    with col_feat:
        with st.expander("Satellite features (pipeline JSON)"):
            st.json(sat_feat)
    with col_tf:
        with st.expander("Temporal features (pipeline JSON)"):
            st.json(tf)

    with st.expander("Limitations and measurement provenance"):
        for lim in result.get("limitations", []):
            st.write(f"• {lim}")
        st.caption(result.get("measurement_provenance", ""))

# ─── Evidence Pipeline Status ─────────────────────────────────────────────
with st.container(border=True):
    st.subheader("📋 Evidence Pipeline Status")
    all_evidence = st.session_state.get("wildlife_evidence", {})
    sat_records  = [r for r in all_evidence.values() if r.get("kind") == "satellite"]

    col_m, col_l = st.columns([2, 3])
    col_m.metric("Satellite evidence records", len(sat_records))
    if sat_records:
        with col_l:
            st.success(
                "Satellite evidence is registered. Navigate to **Reports** "
                "to generate the combined multi-modal assessment."
            )
        for r in sat_records:
            ts = str(r.get("added_at", ""))[:19]
            st.caption(f"• **{r['source']}** — registered {ts}")
    else:
        with col_l:
            st.info(
                "Run the analysis above to register satellite evidence. "
                "Once registered it will contribute to fusion score, "
                "agreement score, anomaly score, ecosystem score, and MEHS."
            )
    st.page_link(
        "wildlife_reports.py",
        label="Open Reports →",
        icon=":material/assignment:",
    )

# ─── Evidence Download ────────────────────────────────────────────────────
with st.expander("⬇️ Download satellite evidence (JSON)"):
    export = {
        "ecomrv_satellite_evidence": {
            "measurement_provenance":  result.get("measurement_provenance"),
            "temporal_features":       tf,
            "temporal_ndvi_series":    series,
            "temporal_anomaly":        anomaly,
            "changes": [
                {k: v for k, v in ch.items() if k != "change_raster_path"}
                for ch in changes
            ],
            "limitations":             result.get("limitations"),
            "scenes_provenance": [
                {k: v for k, v in sc.items() if k in (
                    "original_filename", "acquisition_date", "acquisition_datetime",
                    "tile_id", "crs", "utm_zone", "resolution_m", "raster_dimensions",
                    "cloud_cover_pct", "processing_level", "band_members",
                    "mean_ndvi", "median_ndvi", "std_ndvi", "min_ndvi", "max_ndvi",
                    "valid_pixels", "vegetation_coverage_pct",
                    "low_ndvi_pct", "high_ndvi_pct",
                )}
                for sc in scenes
            ],
        }
    }
    st.download_button(
        "📥 Download satellite_ndvi_evidence.json",
        data=json.dumps(export, indent=2, default=str),
        file_name="satellite_ndvi_evidence.json",
        mime="application/json",
        use_container_width=True,
    )
