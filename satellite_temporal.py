"""Real Sentinel-2 L2A SAFE-ZIP and SAFE-folder NDVI and temporal-change processing.

The reader deliberately works from B04/B08 JP2 assets inside SAFE archives or folders;
it never derives NDVI from display imagery.  Rasterio/GDAL's virtual file
system keeps the input archives/folders unmodified and windowed reads bound memory use.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

from ecomrv_config import NDVI_CHANGE_THRESHOLDS

SAFE_ZIP_PATTERN = "S2*_MSIL2A_*SAFE.zip"
SAFE_FOLDER_PATTERN = "S2*_MSIL2A_*SAFE"
_DATE = re.compile(r"MSIL2A_(\d{8}T\d{6})")
_TILE = re.compile(r"_T(\d{2}[A-Z]{3})_")


def get_default_safe_search_locations() -> list[Path]:
    """Return all sensible default locations where Sentinel-2 SAFE archives may be located."""
    candidates = [
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
        Path.cwd(),
        Path.cwd() / "mrv (3)" / "mrv",
        Path.cwd() / "mrv",
        Path.cwd().parent,
        Path.home() / "Downloads",
        Path("data") / "satellite",
        Path(__file__).resolve().parent / "data" / "satellite",
        Path(__file__).resolve().parent.parent / "data" / "satellite",
        Path(__file__).resolve().parent.parent / "satellite",
    ]
    seen: set[Path] = set()
    result: list[Path] = []
    for p in candidates:
        try:
            rp = p.resolve()
            if rp not in seen and rp.exists() and rp.is_dir():
                seen.add(rp)
                result.append(rp)
        except Exception:
            pass
    return result


def discover_safe_zips(locations: list[str | Path] | None = None) -> list[Path]:
    """Discover, de-duplicate, and name-sort Sentinel-2 L2A SAFE archives (ZIP or folder).
    
    Handles nested SAFE folder structures where the outer folder contains an inner folder
    with the same name (common when extracting ZIP archives).
    """
    default_locs = get_default_safe_search_locations()
    search_locs = list(locations) if locations else []
    for loc in default_locs:
        if loc not in search_locs:
            search_locs.append(loc)

    found: dict[Path, Path] = {}
    for location in search_locs:
        folder = Path(location)
        if folder.is_dir():
            # Discover ZIP archives
            for path in folder.glob(SAFE_ZIP_PATTERN):
                found[path.resolve()] = path.resolve()
            # Discover SAFE folders (directories ending with .SAFE)
            for path in folder.glob(SAFE_FOLDER_PATTERN):
                if path.is_dir():
                    # Check for nested SAFE structure (outer folder contains inner folder with same name)
                    inner_safe = path / path.name
                    if inner_safe.is_dir() and (inner_safe / "MTD_MSIL2A.xml").exists():
                        # Use the inner folder which contains the actual SAFE structure
                        found[inner_safe.resolve()] = inner_safe.resolve()
                    elif (path / "MTD_MSIL2A.xml").exists():
                        # Direct SAFE structure
                        found[path.resolve()] = path.resolve()
    return sorted(found.values(), key=lambda path: (scene_acquisition_datetime(path), path.name))


def scene_acquisition_datetime(path: str | Path) -> datetime:
    match = _DATE.search(Path(path).name)
    if not match:
        raise ValueError(f"Not a recognisable Sentinel-2 L2A SAFE archive: {Path(path).name}")
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def _find_member(names: list[str], suffix: str) -> str:
    candidates = [name for name in names if name.endswith(suffix)]
    if len(candidates) != 1:
        raise ValueError(f"SAFE archive must contain exactly one {suffix}; found {len(candidates)}.")
    return candidates[0]


def _find_member_folder(folder: Path, suffix: str) -> str:
    """Find a file with given suffix in a SAFE folder structure."""
    candidates = list(folder.rglob(f"*{suffix}"))
    if len(candidates) != 1:
        raise ValueError(f"SAFE folder must contain exactly one {suffix}; found {len(candidates)}.")
    return str(candidates[0].relative_to(folder))


def _xml_values(blob: bytes, wanted: set[str]) -> dict[str, list[str]]:
    root = ET.fromstring(blob)
    values = {key: [] for key in wanted}
    for node in root.iter():
        key = node.tag.rsplit("}", 1)[-1]
        if key in values and node.text and node.text.strip():
            values[key].append(node.text.strip())
    return values


def inspect_safe_archive(path: str | Path) -> dict[str, Any]:
    """Validate SAFE layout and return provenance/metadata without reading pixels.
    
    Supports both ZIP archives and SAFE folder structures.
    """
    path = Path(path)
    is_zip = path.is_file() and path.suffix == ".zip"
    
    try:
        if is_zip:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                metadata_member = _find_member(names, "MTD_MSIL2A.xml")
                tile_metadata_member = _find_member(names, "MTD_TL.xml")
                b04 = _find_member(names, "_B04_10m.jp2")
                b08 = _find_member(names, "_B08_10m.jp2")
                product = _xml_values(archive.read(metadata_member), {"PRODUCT_URI", "PROCESSING_LEVEL", "PRODUCT_TYPE", "Cloud_Coverage_Assessment"})
                tile = _xml_values(archive.read(tile_metadata_member), {"HORIZONTAL_CS_CODE", "NROWS", "NCOLS", "ULX", "ULY", "NODATA_VALUE"})
        else:
            # SAFE folder structure
            if not path.is_dir():
                raise ValueError(f"Path is neither a ZIP file nor a directory: {path}")
            metadata_path = path / "MTD_MSIL2A.xml"
            if not metadata_path.exists():
                raise ValueError(f"SAFE folder missing MTD_MSIL2A.xml: {path}")
            b04 = _find_member_folder(path, "_B04_10m.jp2")
            b08 = _find_member_folder(path, "_B08_10m.jp2")
            # Find MTD_TL.xml in GRANULE subfolder
            tile_metadata_path = None
            for granule_folder in path.glob("GRANULE/*/MTD_TL.xml"):
                tile_metadata_path = granule_folder
                break
            if not tile_metadata_path:
                raise ValueError(f"SAFE folder missing MTD_TL.xml in GRANULE: {path}")
            product = _xml_values(metadata_path.read_bytes(), {"PRODUCT_URI", "PROCESSING_LEVEL", "PRODUCT_TYPE", "Cloud_Coverage_Assessment"})
            tile = _xml_values(tile_metadata_path.read_bytes(), {"HORIZONTAL_CS_CODE", "NROWS", "NCOLS", "ULX", "ULY", "NODATA_VALUE"})
    except (OSError, zipfile.BadZipFile, ET.ParseError) as exc:
        raise ValueError(f"Cannot read SAFE archive/folder {path.name}: {exc}") from exc
    tile_match = _TILE.search(path.name)
    acquired = scene_acquisition_datetime(path)
    return {
        "source": str(path), "original_filename": path.name,
        "is_zip": is_zip,
        "sha256": _sha256(path), "acquisition_datetime": acquired.isoformat(), "acquisition_date": acquired.date().isoformat(),
        "tile_id": tile_match.group(1) if tile_match else None,
        "product_type": (product["PRODUCT_TYPE"] or ["S2MSI2A"])[0],
        "processing_level": (product["PROCESSING_LEVEL"] or ["Level-2A"])[0],
        "cloud_cover_pct": _number_or_none((product["Cloud_Coverage_Assessment"] or [None])[0]),
        "crs": (tile["HORIZONTAL_CS_CODE"] or [None])[0],
        "utm_zone": _utm_zone((tile["HORIZONTAL_CS_CODE"] or [""])[0]),
        "resolution_m": 10, "raster_dimensions": {"width": _integer_at(tile["NCOLS"], 0), "height": _integer_at(tile["NROWS"], 0)},
        "nodata": _number_or_none((tile["NODATA_VALUE"] or [None])[0]),
        "band_members": {"B04_red": b04, "B08_nir": b08},
        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def analyze_safe_archives(paths: list[str | Path], output_dir: str | Path | None = None,
                          thresholds: dict[str, float] | None = None, sample_blocks: int = 4) -> dict[str, Any]:
    """Calculate real per-scene NDVI and only compare geometrically overlapping scenes.
    
    Args:
        paths: List of SAFE archive paths (ZIP or folder)
        output_dir: Optional output directory for NDVI rasters
        thresholds: Optional NDVI change thresholds
        sample_blocks: Process every Nth block (1=all, 4=faster, 8=fastest). Default 4 for speed.
    """
    scenes = sorted((inspect_safe_archive(path) for path in paths), key=lambda item: item["acquisition_datetime"])
    if not scenes:
        raise ValueError("No Sentinel-2 Level-2A SAFE archives (ZIP or folder) were discovered.")
    if len({scene["source"] for scene in scenes}) != len(scenes):
        raise ValueError("Duplicate Sentinel-2 archive/folder paths are not allowed.")
    for scene in scenes:
        scene.update(_calculate_scene_ndvi(scene, output_dir, sample_blocks=sample_blocks))
    thresholds = {**NDVI_CHANGE_THRESHOLDS, **(thresholds or {})}
    changes = _temporal_changes(scenes, output_dir, thresholds, sample_blocks=sample_blocks)
    ndvi_values = [scene["mean_ndvi"] for scene in scenes]
    current, historical = ndvi_values[-1], ndvi_values[:-1]
    baseline = float(np.median(historical)) if historical else None
    variability = float(np.std(ndvi_values)) if len(ndvi_values) > 1 else 0.0
    latest_change = changes[-1].get("mean_change") if changes else None
    overall_pixel_change = _first_available_change(changes, "overall")
    one_tile = len({scene["tile_id"] for scene in scenes}) == 1 and scenes[0]["tile_id"] is not None
    comparable_series = one_tile and len({scene["acquisition_date"] for scene in scenes}) >= 3
    overall_scene_change = (scenes[-1]["mean_ndvi"] - scenes[0]["mean_ndvi"]
                            if one_tile and len(scenes) >= 2 else None)
    overall_change = overall_pixel_change if overall_pixel_change is not None else overall_scene_change
    overall_change_method = ("aligned pixel comparison" if overall_pixel_change is not None
                             else "difference between first and latest scene mean NDVI"
                             if overall_scene_change is not None else None)
    anomaly = (_temporal_anomaly(current, historical, latest_change, variability) if comparable_series else
               {"anomaly_score": None, "is_anomaly": None,
                "reason": "A same-tile series with at least three distinct acquisition dates is required; mixed tiles/dates were not combined.",
                "historical_baseline": None, "current_value": None})
    return {
        "measurement_provenance": "Calculated from Sentinel-2 Level-2A B04 (red) and B08 (NIR) JP2 bands in supplied SAFE archives (ZIP or folder).",
        "date_count": len(scenes), "unique_date_count": len({scene["acquisition_date"] for scene in scenes}),
        "first_date": scenes[0]["acquisition_date"], "last_date": scenes[-1]["acquisition_date"],
        "temporal_ndvi_series": [{key: scene[key] for key in ("acquisition_date", "tile_id", "mean_ndvi", "median_ndvi", "std_ndvi", "valid_pixels")} for scene in scenes],
        "scenes": scenes, "changes": changes, "thresholds": thresholds,
        "temporal_features": {"date_count": len(scenes), "first_date": scenes[0]["acquisition_date"], "last_date": scenes[-1]["acquisition_date"],
                              "mean_ndvi": float(np.mean(ndvi_values)), "latest_ndvi": current,
                              "overall_ndvi_change": overall_change,
                              "overall_change_method": overall_change_method,
                              "mean_change": _mean_available(changes, "mean_change"), "max_decline": _min_available(changes, "mean_change"),
                              "max_increase": _max_available(changes, "mean_change"), "temporal_variability": variability,
                              "anomaly_score": anomaly["anomaly_score"]},
        "temporal_anomaly": anomaly,
        "limitations": ["Project/configuration thresholds are not scientifically validated deforestation thresholds.",
                        "Different Sentinel-2 tiles or non-overlapping scenes are not compared at pixel level.",
                        "Mixed tiles are retained as separate observations and are not collapsed into a temporal baseline or fusion score.",
                        "NDVI differences can reflect clouds, atmosphere, seasonality, viewing geometry, or processing effects; further verification is required."],
    }


def _calculate_scene_ndvi(scene: dict[str, Any], output_dir: str | Path | None, sample_blocks: int = 1) -> dict[str, Any]:
    """Calculate NDVI from B04/B08 bands with optional block sampling for speed.
    
    Args:
        scene: Scene metadata dictionary
        output_dir: Optional output directory for NDVI raster
        sample_blocks: Process every Nth block (1 = all blocks, 2 = every 2nd block, etc.)
                      Higher values = faster but less complete processing.
    """
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:
        raise RuntimeError("Rasterio is required for SAFE JP2 processing. Install dependencies from requirements.txt.") from exc
    is_zip = scene.get("is_zip", True)
    red_path = _vsi_zip_path(scene["source"], scene["band_members"]["B04_red"], is_zip=is_zip)
    nir_path = _vsi_zip_path(scene["source"], scene["band_members"]["B08_nir"], is_zip=is_zip)
    with rasterio.open(red_path) as red, rasterio.open(nir_path) as nir:
        if (red.width, red.height, red.transform, red.crs) != (nir.width, nir.height, nir.transform, nir.crs):
            raise ValueError(f"B04/B08 are not spatially compatible in {scene['original_filename']}.")
        scene.update({"crs": red.crs.to_string() if red.crs else scene["crs"], "raster_dimensions": {"width": red.width, "height": red.height},
                      "nodata": red.nodata if red.nodata is not None else nir.nodata, "transform": tuple(red.transform)})
        profile = red.profile.copy()
        writer = _ndvi_writer(profile, output_dir, scene) if output_dir and sample_blocks == 1 else None
        count = low = high = vegetation = 0
        total = total_sq = 0.0
        minimum, maximum = float("inf"), float("-inf")
        samples: list[np.ndarray] = []
        block_count = 0
        total_blocks = sum(1 for _ in red.block_windows(1))
        try:
            for idx, (_, window) in enumerate(red.block_windows(1)):
                # Skip blocks based on sampling rate
                if idx % sample_blocks != 0:
                    continue
                block_count += 1
                r, n = red.read(1, window=window, masked=True), nir.read(1, window=window, masked=True)
                valid = ~(np.ma.getmaskarray(r) | np.ma.getmaskarray(n)) & np.isfinite(r) & np.isfinite(n) & ((r + n) != 0)
                ndvi = np.full(r.shape, np.nan, dtype=np.float32)
                np.divide(n.astype(np.float32) - r.astype(np.float32), n.astype(np.float32) + r.astype(np.float32), out=ndvi, where=valid)
                values = ndvi[valid]
                if values.size:
                    count += values.size; total += float(values.sum()); total_sq += float(np.square(values).sum())
                    minimum = min(minimum, float(values.min())); maximum = max(maximum, float(values.max()))
                    low += int((values < 0.2).sum()); high += int((values >= 0.6).sum()); vegetation += int((values >= 0.3).sum())
                    if sum(part.size for part in samples) < 1_000_000: samples.append(values[:max(0, 1_000_000 - sum(part.size for part in samples))])
                if writer: writer.write(np.where(np.isfinite(ndvi), ndvi, -9999.0), 1, window=window)
        finally:
            if writer: writer.close()
    if not count:
        raise ValueError(f"No valid B04/B08 pixels in {scene['original_filename']}.")
    values = np.concatenate(samples)
    mean = total / count
    return {"mean_ndvi": mean, "median_ndvi": float(np.median(values)), "min_ndvi": minimum, "max_ndvi": maximum,
            "std_ndvi": max(0.0, total_sq / count - mean * mean) ** .5, "valid_pixels": count,
            "vegetation_coverage_pct": vegetation / count * 100, "low_ndvi_pct": low / count * 100, "high_ndvi_pct": high / count * 100,
            "ndvi_raster_path": str(_ndvi_path(output_dir, scene)) if writer else None}


def _temporal_changes(scenes: list[dict[str, Any]], output_dir: str | Path | None, thresholds: dict[str, float], sample_blocks: int = 1) -> list[dict[str, Any]]:
    # Change maps require same CRS/grid and spatial overlap. Adjacent listed scenes
    # can be separate Sentinel tiles, so unavailable comparisons are explicit.
    output = []
    pairs = [(scenes[index], scenes[index + 1], f"change_{index + 1}_to_{index + 2}") for index in range(len(scenes) - 1)]
    if len(scenes) > 1: pairs.append((scenes[0], scenes[-1], "overall"))
    for older, newer, label in pairs:
        result = {"label": label, "from_date": older["acquisition_date"], "to_date": newer["acquisition_date"], "from_tile": older["tile_id"], "to_tile": newer["tile_id"]}
        if older["tile_id"] != newer["tile_id"] or older["crs"] != newer["crs"] or older["transform"] != newer["transform"] or older["raster_dimensions"] != newer["raster_dimensions"]:
            result.update({"status": "unavailable", "reason": "Scenes are different tiles, CRS, transforms, or dimensions; pixel comparison was not performed."})
        else:
            result.update(_change_same_grid(older, newer, output_dir, thresholds, sample_blocks=sample_blocks)); result["status"] = "available"
        output.append(result)
    return output


def _change_same_grid(older: dict[str, Any], newer: dict[str, Any], output_dir: str | Path | None, thresholds: dict[str, float], sample_blocks: int = 1) -> dict[str, Any]:
    # Read aligned B04/B08 by blocks and calculate NDVI difference from source bands,
    # avoiding a second full NDVI array in memory.
    import rasterio
    old_is_zip = older.get("is_zip", True)
    new_is_zip = newer.get("is_zip", True)
    old_red, old_nir = _vsi_zip_path(older["source"], older["band_members"]["B04_red"], is_zip=old_is_zip), _vsi_zip_path(older["source"], older["band_members"]["B08_nir"], is_zip=old_is_zip)
    new_red, new_nir = _vsi_zip_path(newer["source"], newer["band_members"]["B04_red"], is_zip=new_is_zip), _vsi_zip_path(newer["source"], newer["band_members"]["B08_nir"], is_zip=new_is_zip)
    with rasterio.open(old_red) as a_r, rasterio.open(old_nir) as a_n, rasterio.open(new_red) as b_r, rasterio.open(new_nir) as b_n:
        count = inc = dec = stable = 0; total = 0.0
        # Sampling makes the summary much faster. Only write a GeoTIFF when all
        # blocks are processed; a sampled raster would display as incomplete stripes.
        writer = _change_writer(a_r.profile, output_dir, older, newer) if output_dir and sample_blocks == 1 else None
        for idx, (_, window) in enumerate(a_r.block_windows(1)):
            if idx % sample_blocks:
                continue
            ar, an, br, bn = (ds.read(1, window=window, masked=True).astype(np.float32) for ds in (a_r, a_n, b_r, b_n))
            valid = ~(np.ma.getmaskarray(ar) | np.ma.getmaskarray(an) | np.ma.getmaskarray(br) | np.ma.getmaskarray(bn)) & ((ar + an) != 0) & ((br + bn) != 0)
            delta = np.full(ar.shape, np.nan, dtype=np.float32)
            np.divide(bn - br, bn + br, out=delta, where=valid); before = np.zeros(ar.shape, dtype=np.float32)
            np.divide(an - ar, an + ar, out=before, where=valid); delta -= before
            values = delta[valid]
            count += values.size; total += float(values.sum()); dec += int((values <= thresholds["decrease"]).sum()); inc += int((values >= thresholds["increase"]).sum()); stable += int((np.abs(values) <= thresholds["stable"]).sum())
            if writer:
                writer.write(np.where(np.isfinite(delta), delta, -9999.0), 1, window=window)
        if writer:
            writer.close()
    return {"mean_change": total / count if count else None, "valid_pixels": count, "vegetation_decrease_pixels": dec,
            "vegetation_increase_pixels": inc, "stable_pixels": stable, "thresholds": thresholds,
            "change_raster_path": str(_change_path(output_dir, older, newer)) if writer else None,
            "interpretation": "Vegetation-change signal calculated from aligned NDVI; it is not a deforestation or logging determination."}


def _temporal_anomaly(current: float, history: list[float], change: float | None, variability: float) -> dict[str, Any]:
    if len(history) < 2:
        return {"anomaly_score": None, "is_anomaly": None, "reason": "At least three temporally comparable observations are required.", "historical_baseline": history or None, "current_value": current}
    baseline = float(np.median(history)); mad = float(np.median(np.abs(np.asarray(history) - baseline))); scale = max(1.4826 * mad, .01)
    z = abs(current - baseline) / scale; score = min(100.0, z / 3 * 100)
    return {"anomaly_score": score, "is_anomaly": z >= 3, "reason": "Robust deviation from preceding scene-level NDVI values; not a determination of illegal activity.", "historical_baseline": baseline, "current_value": current, "latest_change": change, "temporal_variability": variability}


def _vsi_zip_path(archive: str, member: str, is_zip: bool = True) -> str:
    """Generate rasterio-compatible path for SAFE archive or folder.
    
    For ZIP: uses /vsizip/ virtual file system.
    For folder: uses direct path.
    """
    archive_path = Path(archive).resolve()
    if is_zip:
        return f"/vsizip/{archive_path.as_posix()}/{member}"
    else:
        return str(archive_path / member)
def _sha256(path: Path) -> str:
    """Calculate SHA256 hash for a file or folder.
    
    For folders, hashes the MTD_MSIL2A.xml file as a stable identifier.
    """
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    elif path.is_dir():
        # For folders, hash the metadata XML as a stable identifier
        metadata_path = path / "MTD_MSIL2A.xml"
        if metadata_path.exists():
            with metadata_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
        else:
            # Fallback: hash the directory name and structure
            digest.update(str(path).encode())
    else:
        digest.update(str(path).encode())
    return digest.hexdigest()
def _number_or_none(value: str | None):
    try: return float(value) if value is not None else None
    except ValueError: return None
def _integer_at(values: list[str], index: int): return int(values[index]) if len(values) > index else None
def _utm_zone(crs: str): return crs.replace("EPSG:326", "") if crs.startswith("EPSG:326") else None
def _ndvi_path(output_dir, scene): return Path(output_dir) / f"{Path(scene['original_filename']).stem}_ndvi.tif"
def _ndvi_writer(profile, output_dir, scene):
    import rasterio
    path = _ndvi_path(output_dir, scene); path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(driver="GTiff", dtype="float32", count=1, nodata=-9999.0, compress="deflate")
    return rasterio.open(path, "w", **profile)
def _change_path(output_dir, older, newer): return Path(output_dir) / f"ndvi_change_{older['acquisition_date']}_to_{newer['acquisition_date']}_{older['tile_id']}.tif"
def _change_writer(profile, output_dir, older, newer):
    import rasterio
    path = _change_path(output_dir, older, newer); path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(driver="GTiff", dtype="float32", count=1, nodata=-9999.0, compress="deflate")
    return rasterio.open(path, "w", **profile)
def _first_available_change(rows, label): return next((row.get("mean_change") for row in rows if row["label"] == label and row.get("status") == "available"), None)
def _mean_available(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]; return float(np.mean(values)) if values else None
def _min_available(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]; return min(values) if values else None
def _max_available(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]; return max(values) if values else None
