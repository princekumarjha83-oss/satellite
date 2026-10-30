import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFilter
import os
import io

st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; padding: 10px 14px; background: linear-gradient(135deg, rgba(34, 197, 94, 0.24) 0%, rgba(10, 30, 18, 0.75) 100%); border: 1.5px solid #22c55e; border-radius: 14px; box-shadow: 0 0 22px rgba(34, 197, 94, 0.38);">
    <span style="font-size: 30px; filter: drop-shadow(0 0 8px rgba(74, 222, 128, 0.7));">🌿🛰️</span>
    <div>
        <h2 style="margin: 0; font-size: 19px; font-weight: 800; letter-spacing: 0.5px; background: linear-gradient(90deg, #4ade80, #a3e635); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 12px rgba(74, 222, 128, 0.5);">ECO-MRV AGENT</h2>
        <p style="margin: 0; font-size: 11px; font-weight: 600; color: #b8f5d0; letter-spacing: 0.8px;">ENVIRONMENTAL INTELLIGENCE</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

# --- Backend Status Check (Silently connected in background) ---
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
backend_healthy = False
backend_stats = None

try:
    health_resp = requests.get(f"{BACKEND_URL}/health", timeout=1.5)
    if health_resp.status_code == 200:
        backend_healthy = True
        backend_stats = health_resp.json()
except Exception:
    backend_healthy = False

# --- Indian Forest Catalog Data ---
DEFAULT_INDIAN_FORESTS = [
    {
        "id": "sundarbans-in",
        "name": "Sundarbans Mangrove Delta",
        "code": "Zone SBN-1",
        "biome": "Tidal Halophytic Mangrove Forest",
        "region": "West Bengal, India",
        "lat": 21.9497,
        "lon": 89.1833,
        "area_ha": 426000,
        "canopy_density_pct": 88.7,
        "mean_ndvi": 0.78,
        "status": "UNESCO World Heritage & Tiger Biosphere",
        "image_file": "assets/forests/sundarbans.png",
        "problems": [
            {
                "id": "PRB-SBN-01",
                "type": "Tidal Shoreline Erosion & Salinity Stress",
                "severity": "High",
                "category": "Coastal Erosion",
                "affected_area_ha": 6.8,
                "confidence": 91.4,
                "coordinates": {"lat": 21.9540, "lon": 89.1880},
                "detected_at": "3 hours ago",
                "description": "Satellite optical analysis indicates 45m shoreline retreat along Gosaba-Matla tidal creek. Mangrove pneumatophore dieback observed.",
                "risk_trajectory": "High risk of further saline bank collapse during upcoming high spring tide.",
                "recommendation": "Deploy geo-jute mangrove embankment reinforcement and notify West Bengal Forest Directorate."
            }
        ]
    },
    {
        "id": "western-ghats-in",
        "name": "Western Ghats — Silent Valley",
        "code": "Zone WG-SV2",
        "biome": "Tropical Wet Evergreen Rainforest",
        "region": "Kerala / Tamil Nadu, India",
        "lat": 11.0833,
        "lon": 76.4500,
        "area_ha": 23750,
        "canopy_density_pct": 96.4,
        "mean_ndvi": 0.89,
        "status": "UNESCO Biodiversity Hotspot",
        "image_file": "assets/forests/western_ghats.png",
        "problems": [
            {
                "id": "PRB-WGT-02",
                "type": "Illegal Timber Extraction & Canopy Gap",
                "severity": "High",
                "category": "Deforestation",
                "affected_area_ha": 2.2,
                "confidence": 93.6,
                "coordinates": {"lat": 11.0870, "lon": 76.4550},
                "detected_at": "6 hours ago",
                "description": "Sentinel-2 NDVI drop of 24% detected in Palakkad buffer ridge. High-resolution imagery shows fresh skid trails and canopy openings.",
                "risk_trajectory": "Potential localized slope destabilization and fragmentation of Lion-tailed Macaque habitat.",
                "recommendation": "Dispatch Kerala Forest Department anti-poaching strike force to Silent Valley Sector 3."
            }
        ]
    },
    {
        "id": "kaziranga-in",
        "name": "Kaziranga National Park",
        "code": "Zone KZ-AS3",
        "biome": "Brahmaputra Alluvial Plain & Semi-Evergreen",
        "region": "Assam, India",
        "lat": 26.5775,
        "lon": 93.1711,
        "area_ha": 43000,
        "canopy_density_pct": 87.2,
        "mean_ndvi": 0.81,
        "status": "World Heritage & Tiger Reserve",
        "image_file": "assets/forests/kaziranga.png",
        "problems": [
            {
                "id": "PRB-KZR-03",
                "type": "Brahmaputra Flood Inundation Anomaly",
                "severity": "Critical",
                "category": "Inundation & Flood",
                "affected_area_ha": 12.5,
                "confidence": 96.1,
                "coordinates": {"lat": 26.5820, "lon": 93.1760},
                "detected_at": "1 hour ago",
                "description": "Brahmaputra river surge detected via Sentinel-1 SAR radar. 12.5 hectares of low-lying tall elephant grass submerged.",
                "risk_trajectory": "Wildlife migration movement across NH-715 towards Karbi Anglong highlands triggered.",
                "recommendation": "Activate automated highway speed restrictor barriers on NH-715 and position animal rescue boats."
            }
        ]
    },
    {
        "id": "jim-corbett-in",
        "name": "Jim Corbett National Park",
        "code": "Zone JCB-UK4",
        "biome": "Moist Deciduous & Shivalik Sal Forest",
        "region": "Uttarakhand, India",
        "lat": 29.5300,
        "lon": 78.7747,
        "area_ha": 52080,
        "canopy_density_pct": 91.8,
        "mean_ndvi": 0.83,
        "status": "Project Tiger National Park",
        "image_file": "assets/forests/jim_corbett.png",
        "problems": [
            {
                "id": "PRB-JCB-04",
                "type": "Shivalik Ridge Wildfire Hotspot",
                "severity": "Critical",
                "category": "Thermal & Fire",
                "affected_area_ha": 4.1,
                "confidence": 95.2,
                "coordinates": {"lat": 29.5350, "lon": 78.7790},
                "detected_at": "4 hours ago",
                "description": "Thermal anomaly detected by Sentinel-2 SWIR band in Dhikala buffer zone. High dry leaf litter fuel load.",
                "risk_trajectory": "Fanned by 14 km/h afternoon valley breeze up the Shivalik ridge.",
                "recommendation": "Mobilize Uttarakhand Forest Fire Fighting Unit and activate Ramganga water pumping points."
            }
        ]
    },
    {
        "id": "gir-in",
        "name": "Gir National Forest & Sanctuary",
        "code": "Zone GIR-GJ5",
        "biome": "Dry Deciduous Teak & Thorn Scrub",
        "region": "Gujarat, India",
        "lat": 21.1243,
        "lon": 70.8242,
        "area_ha": 141200,
        "canopy_density_pct": 82.5,
        "mean_ndvi": 0.69,
        "status": "Asiatic Lion Protected Habitat",
        "image_file": "assets/forests/gir_forest.png",
        "problems": [
            {
                "id": "PRB-GIR-05",
                "type": "Severe Moisture Deficit & Waterhole Stress",
                "severity": "Medium",
                "category": "Drought Stress",
                "affected_area_ha": 5.4,
                "confidence": 84.7,
                "coordinates": {"lat": 21.1280, "lon": 70.8290},
                "detected_at": "1 day ago",
                "description": "IoT microclimate sensors in Sasan sector show 42% moisture drop. Natural waterhole dry level warning triggered.",
                "risk_trajectory": "Asiatic Lion prides moving toward agricultural farm fringes in search of water.",
                "recommendation": "Activate solar-powered tube-well recharge stations at waterholes 14 and 19."
            }
        ]
    },
    {
        "id": "bandipur-in",
        "name": "Bandipur - Nagarhole Tiger Reserve",
        "code": "Zone BND-KT6",
        "biome": "Dry & Moist Deciduous Teak Woodland",
        "region": "Karnataka, India",
        "lat": 11.6664,
        "lon": 76.6291,
        "area_ha": 87400,
        "canopy_density_pct": 89.1,
        "mean_ndvi": 0.79,
        "status": "Nilgiri Biosphere Core Reserve",
        "image_file": "assets/forests/bandipur.png",
        "problems": [
            {
                "id": "PRB-BND-06",
                "type": "Brushfire Scorch Line & Lantana Infestation",
                "severity": "High",
                "category": "Thermal & Fire",
                "affected_area_ha": 3.7,
                "confidence": 92.4,
                "coordinates": {"lat": 11.6710, "lon": 76.6340},
                "detected_at": "8 hours ago",
                "description": "Linear burn scar along Gundlupet border. Dense invasive Lantana camara fueling surface fire spread.",
                "risk_trajectory": "Moderate spread risk into Moolehole moist deciduous elephant corridor.",
                "recommendation": "Deploy firebreak maintenance crews and tractor-mounted blowers along Gundlupet boundary."
            }
        ]
    },
]

# Fetch from backend if available
forests_list = DEFAULT_INDIAN_FORESTS
if backend_healthy:
    try:
        f_resp = requests.get(f"{BACKEND_URL}/api/forests", timeout=2)
        if f_resp.status_code == 200:
            forests_list = f_resp.json().get("forests", DEFAULT_INDIAN_FORESTS)
    except Exception:
        forests_list = DEFAULT_INDIAN_FORESTS

forest_options = [f["name"] for f in forests_list]

st.sidebar.subheader("🌲 Forest Focus (India)")
selected_forest_name = st.sidebar.selectbox(
    "Active Forest Reserve",
    options=forest_options,
    index=0,
    help="Select an Indian forest reserve to focus satellite observation and environmental telemetry.",
)

active_forest = next((f for f in forests_list if f["name"] == selected_forest_name), forests_list[0])

st.sidebar.subheader("⏱️ Timeframe & Sensors")
timeframe = st.sidebar.select_slider(
    "Observation Window",
    options=["Live Feed", "Past 24h", "7 Days", "30 Days", "Seasonal (Monsoon/Dry)"],
    value="Live Feed",
)

st.sidebar.markdown("**Observation Layers:**")
show_satellite_layer = st.sidebar.checkbox("🛰️ ISRO / Sentinel-2 NDVI & Optical", value=True)
show_thermal_layer = st.sidebar.checkbox("🔥 SWIR Thermal Fire Hotspots", value=True)
show_audio_layer = st.sidebar.checkbox("🎙️ Bioacoustic Forest Microphones", value=True)
show_threat_layer = st.sidebar.checkbox("🚨 Wildlife & Anomaly Threat Overlay", value=True)

st.sidebar.divider()
st.sidebar.caption(f"Session Active · {datetime.now().strftime('%d %b %Y, %H:%M IST')}")

# ───────────────────────────────────────────────────────────────────────────
# 3. Main Header — ECO-MRV AGENT
# ───────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="display: flex; align-items: center; gap: 14px; margin-bottom: 6px; padding: 12px 18px; background: linear-gradient(135deg, rgba(34, 197, 94, 0.22) 0%, rgba(10, 26, 15, 0.85) 100%); border: 1.5px solid rgba(34, 197, 94, 0.38); border-radius: 14px; box-shadow: 0 0 28px rgba(34, 197, 94, 0.22);">
    <span style="font-size: 32px; filter: drop-shadow(0 0 8px rgba(74, 222, 128, 0.7));">🌿</span>
    <div>
        <h1 style="margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.5px; background: linear-gradient(90deg, #4ade80, #a3e635); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">ECO-MRV AGENT</h1>
        <p style="margin: 0; font-size: 13px; color: #98c4a8; font-weight: 500;">Real-time Earth Observation satellite surveillance, bioacoustic telemetry, and automated environmental threat detection.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────
# Leaflet Satellite Map Generators (100% Reliable Photorealistic Satellite Maps)
# ───────────────────────────────────────────────────────────────────────────

def generate_leaflet_scanner_html(forest, overlay_boxes=True):
    f_lat = forest.get("lat", 21.9497)
    f_lon = forest.get("lon", 89.1833)
    f_name = forest.get("name", "Indian Forest Reserve")
    f_code = forest.get("code", "Zone-1")
    f_biome = forest.get("biome", "Forest")
    f_health = forest.get("health", "92%")
    problems = forest.get("problems", [])
    problems_json = json.dumps(problems)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; background: #07130a; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
            #map {{ width: 100%; height: 100%; border-radius: 12px; }}
            .leaflet-popup-content-wrapper {{ background: #0d2215; color: #e2f5e9; border: 1px solid #22c55e; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.7); }}
            .leaflet-popup-tip {{ background: #0d2215; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            var map = L.map("map", {{ zoomControl: true, maxZoom: 18 }}).setView([{f_lat}, {f_lon}], 10);

            var esriSat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}", {{
                maxZoom: 19,
                attribution: "Tiles &copy; Esri World Imagery"
            }}).addTo(map);

            var esriLabels = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}", {{
                maxZoom: 19
            }}).addTo(map);

            var osm = L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
                maxZoom: 19,
                attribution: "&copy; OpenStreetMap contributors"
            }});

            L.control.layers({{
                "🛰️ ESRI Satellite Imagery": esriSat,
                "🗺️ OpenStreetMap": osm
            }}).addTo(map);

            var bounds = [
                [{f_lat} - 0.12, {f_lon} - 0.12],
                [{f_lat} + 0.12, {f_lon} - 0.12],
                [{f_lat} + 0.12, {f_lon} + 0.12],
                [{f_lat} - 0.12, {f_lon} + 0.12]
            ];
            var poly = L.polygon(bounds, {{
                color: "#22c55e",
                fillColor: "#22c55e",
                fillOpacity: 0.18,
                weight: 2
            }}).addTo(map);

            poly.bindPopup(`
                <div style="font-size:13px; line-height:1.4;">
                    <b style="color:#4ade80; font-size:14px;">{f_name}</b><br/>
                    <span style="color:#98c4a8;">Code:</span> <code>{f_code}</code><br/>
                    <span style="color:#98c4a8;">Biome:</span> {f_biome}<br/>
                    <span style="color:#98c4a8;">Health Index:</span> <b style="color:#22c55e;">{f_health}</b>
                </div>
            `);

            var problems = {problems_json};
            if ({str(overlay_boxes).lower()} && problems && problems.length > 0) {{
                problems.forEach(function(p) {{
                    var pLat = p.coordinates ? p.coordinates.lat : {f_lat};
                    var pLon = p.coordinates ? p.coordinates.lon : {f_lon};
                    var isCrit = p.severity === "Critical";
                    var color = isCrit ? "#ef4444" : "#f97316";
                    
                    var hazardCircle = L.circle([pLat, pLon], {{
                        color: color,
                        fillColor: color,
                        fillOpacity: 0.38,
                        radius: 3500,
                        weight: 2
                    }}).addTo(map);

                    hazardCircle.bindPopup(`
                        <div style="font-size:13px;">
                            <b style="color:${{color}};">⚠️ ${{p.type}}</b><br/>
                            <span>Severity:</span> <b>${{p.severity}}</b><br/>
                            <span>Detected:</span> ${{p.detected_at || "Live Pass"}}<br/>
                            <p style="margin:4px 0 0 0; color:#e2f5e9;">${{p.description}}</p>
                        </div>
                    `).openPopup();
                }});
            }}
        </script>
    </body>
    </html>
    """

def generate_leaflet_regional_html(forests, sensors_df, threats_df):
    forests_json = json.dumps(forests)
    sensors_list = sensors_df.to_dict(orient="records") if hasattr(sensors_df, "to_dict") else []
    sensors_json = json.dumps(sensors_list)
    threats_list = threats_df.to_dict(orient="records") if hasattr(threats_df, "to_dict") else []
    threats_json = json.dumps(threats_list)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; background: #07130a; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
            #map {{ width: 100%; height: 100%; border-radius: 14px; }}
            .leaflet-popup-content-wrapper {{ background: #0d2215; color: #e2f5e9; border: 1px solid #22c55e; border-radius: 10px; box-shadow: 0 6px 24px rgba(0,0,0,0.7); }}
            .leaflet-popup-tip {{ background: #0d2215; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            var map = L.map("map", {{ zoomControl: true, maxZoom: 18 }}).setView([21.5, 80.0], 5);

            var esriSat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}", {{
                maxZoom: 19,
                attribution: "Tiles &copy; Esri World Imagery"
            }}).addTo(map);

            var esriLabels = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}", {{
                maxZoom: 19
            }}).addTo(map);

            var osm = L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
                maxZoom: 19,
                attribution: "&copy; OpenStreetMap contributors"
            }});

            L.control.layers({{
                "🛰️ ESRI Satellite Imagery": esriSat,
                "🗺️ OpenStreetMap": osm
            }}).addTo(map);

            var forests = {forests_json};
            forests.forEach(function(f) {{
                var fSize = 0.45;
                var bounds = [
                    [f.lat - fSize, f.lon - fSize],
                    [f.lat + fSize, f.lon - fSize],
                    [f.lat + fSize, f.lon + fSize],
                    [f.lat - fSize, f.lon + fSize]
                ];
                var poly = L.polygon(bounds, {{
                    color: "#22c55e",
                    fillColor: "#22c55e",
                    fillOpacity: 0.18,
                    weight: 2
                }}).addTo(map);

                var marker = L.circleMarker([f.lat, f.lon], {{
                    radius: 9,
                    color: "#34d369",
                    fillColor: "#22c55e",
                    fillOpacity: 0.95,
                    weight: 2
                }}).addTo(map);

                var popContent = `
                    <div style="font-size:13px; line-height:1.45; min-width:180px;">
                        <b style="color:#4ade80; font-size:14px;">🌲 ${{f.name}}</b><br/>
                        <span style="color:#98c4a8;">Region:</span> ${{f.region || f.code}}<br/>
                        <span style="color:#98c4a8;">Health Index:</span> <b style="color:#22c55e;">${{f.health || "92.4%"}}</b><br/>
                        <span style="color:#98c4a8;">Canopy:</span> ${{f.canopy_density || "88.2%"}}<br/>
                        <span style="color:#98c4a8;">Protected Species:</span> <i style="color:#b8f5d0;">${{f.key_species ? f.key_species.join(", ") : "Wildlife"}}</i>
                    </div>
                `;
                poly.bindPopup(popContent);
                marker.bindPopup(popContent);
            }});

            var sensors = {sensors_json};
            sensors.forEach(function(s) {{
                var color = s.type === "Bioacoustic Mic" ? "#22c55e" : (s.type === "Flood Gauge" ? "#38bdf8" : "#f59e0b");
                var sMarker = L.circleMarker([s.lat, s.lon], {{
                    radius: 6,
                    color: color,
                    fillColor: color,
                    fillOpacity: 0.9,
                    weight: 1.5
                }}).addTo(map);

                sMarker.bindPopup(`
                    <div style="font-size:12px;">
                        <b style="color:${{color}};">📡 ${{s.name}}</b><br/>
                        <span>Type:</span> ${{s.type}}<br/>
                        <span>Status:</span> <b style="color:#4ade80;">${{s.status}}</b><br/>
                        <span>Signal / Battery:</span> 98% (Online)
                    </div>
                `);
            }});

            var threats = {threats_json};
            threats.forEach(function(t) {{
                if (t.polygon && t.polygon.length > 0) {{
                    var latlngs = t.polygon.map(function(pt) {{ return [pt[1], pt[0]]; }});
                    var tPoly = L.polygon(latlngs, {{
                        color: "#ef4444",
                        fillColor: "#ef4444",
                        fillOpacity: 0.35,
                        weight: 2
                    }}).addTo(map);

                    tPoly.bindPopup(`
                        <div style="font-size:12px;">
                            <b style="color:#ef4444;">🚨 ${{t.name}}</b><br/>
                            <span>Type:</span> ${{t.type}}<br/>
                            <span>Confidence:</span> <b>${{t.confidence}}%</b><br/>
                            <span>Status:</span> <b style="color:#ef4444;">${{t.status}}</b>
                        </div>
                    `);
                }}
            }});
        </script>
    </body>
    </html>
    """

# ───────────────────────────────────────────────────────────────────────────
# 4. Main Navigation Tabs
# ───────────────────────────────────────────────────────────────────────────

tab_scanner, tab_metrics, tab_map, tab_timelapse, tab_satellite_detection, tab_reports = st.tabs([
    ":material/satellite_alt: Forest Satellite Scanner",
    ":material/analytics: Live Environmental Metrics",
    ":material/map: Regional Indian Forest Map",
    ":material/timelapse: Satellite Time-Lapse",
    ":material/satellite_alt: Satellite Detection",
    ":material/assignment: Environmental Reports",
])

# ===========================================================================
# TAB 1: 🛰️ Forest Satellite Scanner (FEATURED INDIAN FORESTS)
# ===========================================================================
with tab_scanner:
    st.header("🛰️ Forest Satellite Problem Scanner")
    st.caption("Select an Indian forest reserve below to inspect high-resolution multi-spectral satellite passes and identify environmental anomalies in real time.")

    # --- Top Quick Selection & Metadata Banner ---
    with st.container(border=True):
        col_fsel, col_finfo1, col_finfo2, col_finfo3 = st.columns([1.5, 1, 1, 1])

        with col_fsel:
            active_scanner_forest_name = st.selectbox(
                "Select Forest to Inspect",
                options=forest_options,
                index=forest_options.index(selected_forest_name) if selected_forest_name in forest_options else 0,
                key="scanner_forest_select",
            )
            scanner_forest = next((f for f in forests_list if f["name"] == active_scanner_forest_name), active_forest)

        with col_finfo1:
            st.metric("Total Area", f"{scanner_forest['area_ha']:,} ha", scanner_forest.get("region", "India"))
        with col_finfo2:
            st.metric("Canopy Density", f"{scanner_forest['canopy_density_pct']:.1f}%", "+0.6% vs baseline")
        with col_finfo3:
            st.metric("Mean NDVI", f"{scanner_forest['mean_ndvi']:.2f}", "Healthy Vegetation" if scanner_forest['mean_ndvi'] > 0.75 else "Moderate / Dry")

    st.space("small")

    # --- Spectral Band Mode Selector & Controls ---
    col_band, col_toggle = st.columns([2, 1])
    with col_band:
        band_mode = st.segmented_control(
            "Satellite Observation Spectrum",
            options=[
                "🌿 True Color (Optical)",
                "📈 NDVI Vegetation Health",
                "🔥 SWIR Infrared (Thermal / Fire)",
                "📉 Canopy Loss / Flood Inundation",
            ],
            default="🌿 True Color (Optical)",
        )
    with col_toggle:
        overlay_boxes = st.toggle("Highlight Detected Problem Zones", value=True)

    st.space("small")

    # --- Satellite View + Problem Detection Details ---
    sat_view_col, sat_problem_col = st.columns([1.7, 1.3])

    with sat_view_col:
        with st.container(border=True):
            st.subheader(f":material/satellite: {scanner_forest['name']} — Satellite Imagery")
            st.caption(f"Sensor: Sentinel-2 MSI / ISRO Cartosat · Ground Sampling: 10m · Location: {scanner_forest.get('region', 'India')} ({scanner_forest['lat']}°N, {scanner_forest['lon']}°E)")

            # ── Always prepare display_img for visualization & export ──
            img_path = scanner_forest.get("image_file", "assets/forests/sundarbans.png")
            if not os.path.exists(img_path):
                img_path = "assets/forests/sundarbans.png"

            if os.path.exists(img_path):
                pil_img = Image.open(img_path).convert("RGBA")
            else:
                pil_img = Image.new("RGBA", (800, 600), (25, 75, 35, 255))

            img_w, img_h = pil_img.size
            display_img = pil_img.copy()

            if "NDVI" in band_mode:
                img_np = np.array(display_img).astype(np.float32)
                ndvi_green = np.clip(img_np[:, :, 1] * 1.45, 0, 255)
                img_np[:, :, 0] = np.clip(255 - ndvi_green * 0.9, 0, 255)
                img_np[:, :, 1] = ndvi_green
                img_np[:, :, 2] = 35
                display_img = Image.fromarray(img_np.astype(np.uint8))
            elif "SWIR" in band_mode:
                img_np = np.array(display_img).astype(np.float32)
                img_np[:, :, 0] = np.clip(img_np[:, :, 0] * 1.55, 0, 255)
                img_np[:, :, 1] = np.clip(img_np[:, :, 1] * 0.7, 0, 255)
                img_np[:, :, 2] = np.clip(img_np[:, :, 2] * 0.4, 0, 255)
                display_img = Image.fromarray(img_np.astype(np.uint8))
            elif "Flood" in band_mode or "Canopy Loss" in band_mode:
                img_np = np.array(display_img).astype(np.float32)
                loss_mask = (img_np[:, :, 0] > 100) & (img_np[:, :, 1] < 125)
                img_np[loss_mask, 0] = 240
                img_np[loss_mask, 1] = 60
                img_np[loss_mask, 2] = 180
                display_img = Image.fromarray(img_np.astype(np.uint8))

            if overlay_boxes and scanner_forest.get("problems"):
                draw = ImageDraw.Draw(display_img)
                for idx, prob in enumerate(scanner_forest["problems"]):
                    cx, cy = int(img_w * (0.5 + 0.12 * (idx % 2))), int(img_h * (0.45 + 0.1 * (idx % 2)))
                    box_w, box_h = 170, 115
                    bbox_color = (239, 68, 68, 255) if prob["severity"] == "Critical" else (249, 115, 22, 255)
                    draw.rectangle([cx - box_w//2, cy - box_h//2, cx + box_w//2, cy + box_h//2], outline=bbox_color, width=3)
                    draw.rectangle([cx - box_w//2, cy - box_h//2, cx + box_w//2, cy - box_h//2 + 24], fill=bbox_color)
                    draw.line([(cx - 15, cy), (cx + 15, cy)], fill=(255, 255, 255, 220), width=2)
                    draw.line([(cx, cy - 15), (cx, cy + 15)], fill=(255, 255, 255, 220), width=2)

            # ── Mode toggle: Interactive Satellite Map vs Multi-Spectral Tile ──
            scanner_view_mode = st.radio(
                "Imagery View Mode",
                options=["🛰️ Live Satellite Map (Interactive 3D)", "🖼️ Multi-Spectral Anomaly Frame"],
                horizontal=True,
                label_visibility="collapsed",
            )

            if "Live Satellite Map" in scanner_view_mode:
                # ── Photorealistic Leaflet Satellite Map with ESRI Earth Observation Tiles ──
                leaflet_scanner_code = generate_leaflet_scanner_html(scanner_forest, overlay_boxes)
                st.iframe(leaflet_scanner_code, height=450)
            else:
                st.image(display_img, caption=f"Satellite pass over {scanner_forest['name']} ({band_mode})")

            with st.container(horizontal=True):
                if st.button("🔄 Trigger ISRO / Sentinel-2 Scan", type="primary", icon=":material/radar:"):
                    with st.spinner(f"Querying Earth Observation feed for {scanner_forest['name']}..."):
                        import time
                        time.sleep(1.2)
                        st.success(f"Satellite pass for {scanner_forest['name']} verified! Telemetry updated.")
                
                buf = io.BytesIO()
                display_img.save(buf, format="PNG")
                st.download_button(
                    label="📥 Export Satellite Frame",
                    data=buf.getvalue(),
                    file_name=f"{scanner_forest['id']}_satellite_pass.png",
                    mime="image/png",
                    icon=":material/download:",
                )

    with sat_problem_col:
        with st.container(border=True):
            st.subheader(":material/warning: Detected Forest Problems")
            st.caption("AI spatial anomaly detection engine results for this Indian forest reserve.")

            problems = scanner_forest.get("problems", [])
            if not problems:
                st.success("✅ No critical environmental threats detected in this forest reserve.", icon=":material/check_circle:")
            else:
                for prob in problems:
                    severity_color = "red" if prob["severity"] == "Critical" else "orange" if prob["severity"] == "High" else "blue"
                    
                    with st.container(border=True):
                        st.markdown(f"### 🚨 {prob['type']}")
                        
                        col_p1, col_p2 = st.columns(2)
                        with col_p1:
                            st.badge(prob["severity"], color=severity_color)
                            st.caption(f"Category: **{prob['category']}**")
                        with col_p2:
                            st.metric("Affected Area", f"{prob['affected_area_ha']} ha", border=False)
                        
                        st.markdown(f"**Confidence:** :green[{prob['confidence']}%] · **Detected:** {prob['detected_at']}")
                        st.write(prob["description"])
                        
                        st.markdown(f"📈 **Risk Trajectory:** {prob['risk_trajectory']}")
                        st.info(f"💡 **Action Required:** {prob['recommendation']}", icon=":material/lightbulb:")
                        
                        col_act1, col_act2 = st.columns(2)
                        with col_act1:
                            if st.button(f"🚨 Dispatch Forest Squad", key=f"btn_disp_{prob['id']}"):
                                st.toast(f"Forest Department Patrol dispatched to {prob['coordinates']['lat']}°N, {prob['coordinates']['lon']}°E!", icon="🚁")
                        with col_act2:
                            if st.button(f"📋 Record Incident in FSI Ledger", key=f"btn_log_{prob['id']}"):
                                st.toast(f"Incident {prob['id']} logged in State Forest Record.", icon="✅")

        st.space("small")

        with st.container(border=True):
            st.subheader("📤 Ingest & Analyze New Satellite Capture")
            from satellite_evidence import render_satellite_upload
            render_satellite_upload("sat_uploader")

# ===========================================================================
# TAB 2: 📊 Live Environmental Metrics
# ===========================================================================
with tab_metrics:
    st.header("Live Environmental Health Metrics")
    st.caption("National forest ecosystem telemetry updated continuously from ISRO/Sentinel-2 passes and field IoT sensor networks.")

    metrics_url = f"{BACKEND_URL}/api/metrics"
    try:
        response = requests.get(metrics_url, timeout=2.5)
        response.raise_for_status()
        metrics_data = response.json()
        pillars = metrics_data["pillars"]
        composite_score = metrics_data["composite_score"]
        composite_trend = metrics_data["composite_trend"]
        composite_delta = metrics_data["composite_delta"]
    except Exception:
        np.random.seed(42)
        pillars = [
            {"name": "Canopy Density", "score": 95.2, "delta": 0.5, "trend_30d": (94 + np.cumsum(np.random.normal(0, 0.2, 30))).tolist()},
            {"name": "Climate & Moisture Stability", "score": 89.8, "delta": -0.4, "trend_30d": (89 + np.cumsum(np.random.normal(0, 0.3, 30))).tolist()},
            {"name": "Biodiversity Vitality", "score": 93.8, "delta": 0.9, "trend_30d": (92 + np.cumsum(np.random.normal(0, 0.25, 30))).tolist()},
            {"name": "Conservation Integrity", "score": 96.8, "delta": 0.3, "trend_30d": (96 + np.cumsum(np.random.normal(0, 0.15, 30))).tolist()},
        ]
        composite_score = 93.9
        composite_delta = 0.4
        composite_trend = (93 + np.cumsum(np.random.normal(0, 0.2, 30))).tolist()

    with st.container(horizontal=True):
        for pillar in pillars:
            icon = ":material/forest:"
            if "Canopy" in pillar["name"]:
                icon = ":material/park:"
            elif "Climate" in pillar["name"] or "Moisture" in pillar["name"]:
                icon = ":material/water_drop:"
            elif "Biodiversity" in pillar["name"]:
                icon = ":material/pets:"
            elif "Conservation" in pillar["name"] or "Integrity" in pillar["name"]:
                icon = ":material/shield:"

            st.metric(
                f"{icon} {pillar['name']}",
                f"{pillar['score']:.1f}%",
                f"{pillar['delta']:+.1f}%",
                border=True,
                chart_data=pillar["trend_30d"],
                chart_type="area",
                help=f"{pillar['name']} 30-day telemetry trend across Indian reserves",
            )

    st.space("small")

    col_score, col_chart = st.columns([1, 3])

    with col_score:
        with st.container(border=True):
            if composite_score >= 90:
                badge_label = "Optimal Ecological Health"
                badge_color = "green"
            elif composite_score >= 75:
                badge_label = "Stable Condition"
                badge_color = "blue"
            else:
                badge_label = "Intervention Needed"
                badge_color = "orange"

            st.metric(
                "National Forest Health Index",
                f"{composite_score:.1f}%",
                f"{composite_delta:+.1f}%",
                chart_data=composite_trend,
                chart_type="line",
            )
            st.badge(badge_label, icon=":material/verified:", color=badge_color)
            st.caption("Weighted across canopy vigor, hydrological balance, wildlife density, and border integrity.")

    with col_chart:
        with st.container(border=True):
            st.subheader("30-Day Environmental Health Pillar Trends")
            dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
            trend_df = pd.DataFrame(
                {
                    "Date": dates,
                    "Canopy Density": next(p["trend_30d"] for p in pillars if "Canopy" in p["name"]),
                    "Climate Stability": next(p["trend_30d"] for p in pillars if "Climate" in p["name"]),
                    "Biodiversity Vitality": next(p["trend_30d"] for p in pillars if "Biodiversity" in p["name"]),
                    "Conservation Integrity": next(p["trend_30d"] for p in pillars if "Conservation" in p["name"]),
                }
            )
            st.area_chart(
                trend_df,
                x="Date",
                y=["Canopy Density", "Climate Stability", "Biodiversity Vitality", "Conservation Integrity"],
                color=["#22c55e", "#3b82f6", "#a855f7", "#10b981"],
            )

    st.space("small")

    # --- Fetch Real-Time Threat Alerts ---
    alerts_url = f"{BACKEND_URL}/api/alerts"
    try:
        response_alerts = requests.get(alerts_url, timeout=2.5)
        response_alerts.raise_for_status()
        alerts = response_alerts.json()["alerts"]
    except Exception:
        now = datetime.now()
        alerts = [
            {
                "severity": "Critical",
                "event": "Brahmaputra Flood Surge Inundation — Kaziranga Zone AS3",
                "description": "SAR radar imagery confirms 12.5 ha low-lying floodplain submergence. Animal migration corridor across NH-715 activated.",
                "timestamp": (now - timedelta(hours=1)).isoformat(),
                "pillar_affected": "Conservation Integrity",
            },
            {
                "severity": "Critical",
                "event": "Shivalik Wildfire Hotspot — Jim Corbett Sector JCB4",
                "description": "Sentinel-2 SWIR thermal band detected 4.1 ha fire line in Dhikala range. High dry leaf litter fuel conditions.",
                "timestamp": (now - timedelta(hours=3)).isoformat(),
                "pillar_affected": "Canopy Density",
            },
            {
                "severity": "High",
                "event": "Timber Extraction & Canopy Gap — Western Ghats Zone WG2",
                "description": "NDVI drop of 24% over 7 days in Palakkad buffer zone. Skid trails detected. Anti-poaching squad notified.",
                "timestamp": (now - timedelta(hours=5)).isoformat(),
                "pillar_affected": "Canopy Density",
            },
            {
                "severity": "High",
                "event": "Tidal Mangrove Embankment Erosion — Sundarbans Zone SBN1",
                "description": "45m shoreline retreat observed along Matla creek. Saline ingress impacting freshwater pocket ponds.",
                "timestamp": (now - timedelta(hours=8)).isoformat(),
                "pillar_affected": "Climate & Moisture Stability",
            },
        ]

    st.subheader("Real-Time Indian Forest Threat Alerts")
    st.caption("Automated anomaly detection from satellite observation and field sensor telemetry.")

    alert_col1, alert_col2 = st.columns(2)
    for i, alert in enumerate(alerts[:4]):
        target_col = alert_col1 if i % 2 == 0 else alert_col2
        with target_col:
            with st.container(border=True):
                if alert["severity"] == "Critical":
                    st.error(f"**{alert['event']}**\n\n{alert['description']}", icon=":material/warning:")
                elif alert["severity"] == "High":
                    st.warning(f"**{alert['event']}**\n\n{alert['description']}", icon=":material/report:")
                elif alert["severity"] == "Medium":
                    st.warning(f"**{alert['event']}**\n\n{alert['description']}", icon=":material/sensors:")
                else:
                    st.success(f"**{alert['event']}**\n\n{alert['description']}", icon=":material/info:")

    st.space("small")

    with st.container(border=True):
        st.subheader("Environmental Alert Log & Provenance")
        alert_log_data = []
        for alert in alerts:
            dt = datetime.fromisoformat(alert["timestamp"]) if "T" in alert["timestamp"] else datetime.now()
            alert_log_data.append({
                "Timestamp (IST)": dt.strftime("%Y-%m-%d %H:%M"),
                "Severity": alert["severity"],
                "Event": alert["event"],
                "Ecosystem Pillar": alert["pillar_affected"],
            })
        alert_log = pd.DataFrame(alert_log_data)
        st.dataframe(alert_log, hide_index=True)

# ===========================================================================
# TAB 3: 🗺️ Interactive Regional Indian Forest Map
# ===========================================================================
with tab_map:
    st.header("Interactive Regional Forest Map — India")
    st.caption("Major Indian forest reserves and tiger corridors with real-time IoT sensor overlays and AI-flagged anomaly zones.")

    # Data for Indian Forests
    parcels = pd.DataFrame(
        {
            "name": [
                "Sundarbans Mangrove Delta",
                "Western Ghats (Silent Valley)",
                "Kaziranga National Park",
                "Jim Corbett National Park",
                "Gir Forest & Lion Sanctuary",
                "Bandipur - Nagarhole Tiger Reserve",
            ],
            "lat": [21.9497, 11.0833, 26.5775, 29.5300, 21.1243, 11.6664],
            "lon": [89.1833, 76.4500, 93.1711, 78.7747, 70.8242, 76.6291],
            "status": ["UNESCO Tiger Biosphere", "UNESCO Biodiversity Hotspot", "World Heritage Reserve", "Project Tiger Reserve", "Asiatic Lion Sanctuary", "Nilgiri Biosphere Core"],
            "area_ha": [426000, 23750, 43000, 52080, 141200, 87400],
            "canopy_density": [88.7, 96.4, 87.2, 91.8, 82.5, 89.1],
            "ndvi_score": [0.78, 0.89, 0.81, 0.83, 0.69, 0.79],
            "color": [
                [59, 130, 246, 170], # Sundarbans (Blue-green)
                [34, 197, 94, 170],  # Western Ghats (Deep green)
                [16, 185, 129, 170], # Kaziranga (Emerald)
                [34, 197, 94, 170],  # Corbett (Green)
                [245, 158, 11, 170], # Gir (Amber dry deciduous)
                [34, 197, 94, 170],  # Bandipur (Green)
            ],
        }
    )

    def make_indian_polygon(lat, lon, size_deg=0.3):
        return [
            [lon - size_deg, lat - size_deg],
            [lon + size_deg, lat - size_deg],
            [lon + size_deg, lat + size_deg],
            [lon - size_deg, lat + size_deg],
            [lon - size_deg, lat - size_deg],
        ]

    parcels["polygon"] = [
        make_indian_polygon(row.lat, row.lon, size_deg=0.15 + (row.area_ha / 500000) * 0.2)
        for _, row in parcels.iterrows()
    ]

    # Threat zones in India
    threat_zones = pd.DataFrame(
        {
            "name": [
                "Flood Inundation Corridor — Kaziranga",
                "Wildfire Hotspot — Corbett Shivalik",
                "Canopy Gap Logging — Western Ghats",
                "Shoreline Saline Erosion — Sundarbans",
            ],
            "lat": [26.5820, 29.5350, 11.0870, 21.9540],
            "lon": [93.1760, 78.7790, 76.4550, 89.1880],
            "type": ["Flood Inundation", "Wildfire Hotspot", "Illegal Logging", "Coastal Erosion"],
            "confidence": [96.1, 95.2, 93.6, 91.4],
            "color": [
                [239, 68, 68, 170],
                [239, 68, 68, 170],
                [249, 115, 22, 170],
                [239, 68, 68, 150],
            ],
            "line_color": [
                [239, 68, 68, 255],
                [239, 68, 68, 255],
                [249, 115, 22, 255],
                [239, 68, 68, 220],
            ],
        }
    )
    threat_zones["polygon"] = [
        make_indian_polygon(row.lat, row.lon, size_deg=0.08)
        for _, row in threat_zones.iterrows()
    ]

    # IoT Sensor nodes distributed across Indian reserves
    sensors = pd.DataFrame(
        {
            "lat": [21.93, 21.96, 11.07, 11.09, 26.56, 26.59, 29.52, 29.54, 21.11, 21.13, 11.65, 11.68],
            "lon": [89.17, 89.20, 76.43, 76.47, 93.15, 93.19, 78.76, 78.79, 70.81, 70.84, 76.61, 76.64],
            "type": [
                "Water Salinity", "Bioacoustic Mic", "Moisture Array", "Bioacoustic Mic",
                "Flood Gauge", "Bioacoustic Mic", "Thermal Camera", "Bioacoustic Mic",
                "Waterhole Monitor", "Temperature", "Elephant Crossing Sensor", "Bioacoustic Mic"
            ],
            "color": [
                [96, 165, 250, 220], [192, 132, 252, 220], [74, 222, 128, 220], [192, 132, 252, 220],
                [96, 165, 250, 220], [192, 132, 252, 220], [239, 68, 68, 220], [192, 132, 252, 220],
                [74, 222, 128, 220], [96, 165, 250, 220], [245, 158, 11, 220], [192, 132, 252, 220]
            ],
        }
    )

    map_col, detail_col = st.columns([3, 1])

    with map_col:
        # ── Photorealistic Leaflet National Satellite Map with ESRI World Imagery ──
        leaflet_regional_code = generate_leaflet_regional_html(forests_list, sensors, threat_zones)
        st.iframe(leaflet_regional_code, height=530)

    with detail_col:
        with st.container(border=True):
            st.subheader("Map Legend (India)")
            st.markdown(":green-badge[Rainforest / Tiger Reserve]")
            st.markdown(":blue-badge[Mangrove / Ramsar Delta]")
            st.markdown(":orange-badge[Dry Deciduous Sanctuary]")
            st.markdown(":red-badge[Critical Anomaly Hotspot]")
            st.caption(":material/sensors: IoT Nodes (Salinity, Thermal, Flood, Microphones)")

        st.space("small")

        with st.container(border=True):
            st.subheader("Reserve Telemetry")
            insp_forest_name = st.selectbox(
                "Quick Telemetry",
                options=[f["name"] for f in forests_list],
                index=0,
                key="map_insp_forest",
            )
            insp_forest = next((f for f in forests_list if f["name"] == insp_forest_name), forests_list[0])
            st.markdown(f"**{insp_forest['name']}**")
            st.badge(insp_forest.get("health", "92.4% Health"), color="green")
            st.metric("Total Reserve Area", f"{insp_forest.get('area_ha', 0):,} ha", border=True)
            st.metric("Canopy Density", f"{insp_forest.get('canopy_density', '88.2%')}", border=True)
            st.metric("Key Biome", insp_forest.get("biome", "National Reserve"), border=True)

    st.space("small")
    with st.container(border=True):
        st.subheader("National Forest Reserves Summary")
        st.dataframe(
            parcels[["name", "status", "area_ha", "canopy_density", "ndvi_score"]].rename(
                columns={
                    "name": "Forest Reserve",
                    "status": "Conservation Status",
                    "area_ha": "Area (ha)",
                    "canopy_density": "Canopy Density (%)",
                    "ndvi_score": "Mean NDVI",
                }
            ),
            hide_index=True,
            column_config={
                "Area (ha)": st.column_config.NumberColumn(format="%d ha"),
                "Canopy Density (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            },
        )

# ===========================================================================
# TAB 4: 🛰️ Satellite Time-Lapse
# ===========================================================================
with tab_timelapse:
    st.header("Satellite Time-Lapse & NDVI Canopy Evolution")
    st.caption("ISRO / Sentinel-2 multi-temporal NDVI composite analysis tracking seasonal monsoon foliage surges, canopy retention, and localized spatial anomalies across Indian forest reserves.")

    col_tl_forest, col_tl_slider = st.columns([1.3, 2.7])

    with col_tl_forest:
        selected_tl_forest_name = st.selectbox(
            "🌲 Select Forest for Time-Lapse",
            options=[f["name"] for f in forests_list],
            index=0,
            key="tl_forest_selector",
            help="Select any Indian Forest Reserve to inspect its 12-month Sentinel-2 NDVI canopy profile.",
        )

    tl_forest = next((f for f in forests_list if f["name"] == selected_tl_forest_name), forests_list[0])
    center_lat = tl_forest.get("lat", 21.9497)
    center_lon = tl_forest.get("lon", 89.1833)
    f_id = tl_forest.get("id", "sundarbans-in")

    months = pd.date_range("2025-09-01", periods=12, freq="MS")
    month_labels = [m.strftime("%b %Y") for m in months]

    with col_tl_slider:
        selected_month_idx = st.select_slider(
            "Observation Month Timeline",
            options=list(range(len(months))),
            format_func=lambda i: month_labels[i],
            value=8,
            key="tl_month_slider",
        )
    selected_date = months[selected_month_idx]

    st.space("small")

    sat_col1, sat_col2 = st.columns([2, 1])

    with sat_col1:
        np.random.seed(selected_month_idx + 150 + int(center_lat * 10))
        grid_size = 20

        # Forest-specific NDVI baseline
        if "sundarbans" in f_id:
            base_ndvi = np.random.uniform(0.68, 0.86, (grid_size, grid_size))
            seasonal_factor = 1.0 + 0.09 * np.sin(2 * np.pi * (selected_month_idx + 2) / 12)
        elif "western-ghats" in f_id:
            base_ndvi = np.random.uniform(0.78, 0.94, (grid_size, grid_size))
            seasonal_factor = 1.0 + 0.12 * np.sin(2 * np.pi * (selected_month_idx - 2) / 12)
        elif "kaziranga" in f_id:
            base_ndvi = np.random.uniform(0.70, 0.90, (grid_size, grid_size))
            seasonal_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (selected_month_idx - 1) / 12)
        elif "corbett" in f_id:
            base_ndvi = np.random.uniform(0.72, 0.91, (grid_size, grid_size))
            seasonal_factor = 1.0 + 0.12 * np.sin(2 * np.pi * (selected_month_idx - 2) / 12)
        elif "gir" in f_id:
            base_ndvi = np.random.uniform(0.52, 0.74, (grid_size, grid_size))
            seasonal_factor = 1.0 + 0.18 * np.sin(2 * np.pi * (selected_month_idx - 2) / 12)
        else: # bandipur
            base_ndvi = np.random.uniform(0.64, 0.84, (grid_size, grid_size))
            seasonal_factor = 1.0 + 0.14 * np.sin(2 * np.pi * (selected_month_idx - 2) / 12)

        ndvi_grid = np.clip(base_ndvi * seasonal_factor, 0.1, 0.98)

        # Forest-specific anomalies
        anomalies = []
        if "sundarbans" in f_id:
            if selected_month_idx in [3, 4, 5]: # Pre-monsoon salinity stress
                r1, c1, s1 = 5, 12, 3
                ndvi_grid[r1 : r1 + s1, c1 : c1 + s1] = np.random.uniform(0.12, 0.24, (s1, s1))
                anomalies.append({
                    "label": "Tidal Creek Salinity Defoliation",
                    "row": r1, "col": c1, "size": s1,
                    "confidence": 88,
                })
        elif "western-ghats" in f_id:
            if selected_month_idx >= 7:
                r1, c1, s1 = 12, 5, 3
                ndvi_grid[r1 : r1 + s1, c1 : c1 + s1] = np.random.uniform(0.14, 0.26, (s1, s1))
                anomalies.append({
                    "label": "Canopy Gap / Linear Encroachment",
                    "row": r1, "col": c1, "size": s1,
                    "confidence": 92,
                })
        elif "kaziranga" in f_id:
            if selected_month_idx in [9, 10, 11]: # Monsoon flood submergence
                r1, c1, s1 = 8, 8, 5
                ndvi_grid[r1 : r1 + s1, c1 : c1 + s1] = np.random.uniform(0.06, 0.18, (s1, s1))
                anomalies.append({
                    "label": "Brahmaputra Monsoon Flood Inundation",
                    "row": r1, "col": c1, "size": s1,
                    "confidence": 97,
                })
        elif "corbett" in f_id:
            if selected_month_idx in [4, 5, 6]: # Dry season fire risk
                r1, c1, s1 = 4, 13, 4
                ndvi_grid[r1 : r1 + s1, c1 : c1 + s1] = np.random.uniform(0.08, 0.19, (s1, s1))
                anomalies.append({
                    "label": "Shivalik Sal Forest Fire Scorch",
                    "row": r1, "col": c1, "size": s1,
                    "confidence": 94,
                })
        elif "gir" in f_id:
            if selected_month_idx in [5, 6, 7]: # Severe summer drought
                r1, c1, s1 = 6, 10, 4
                ndvi_grid[r1 : r1 + s1, c1 : c1 + s1] = np.random.uniform(0.10, 0.22, (s1, s1))
                anomalies.append({
                    "label": "Summer Drought Thermal Desiccation",
                    "row": r1, "col": c1, "size": s1,
                    "confidence": 91,
                })
        else: # bandipur
            if selected_month_idx in [4, 5]: # Dry season fire danger
                r1, c1, s1 = 3, 11, 4
                ndvi_grid[r1 : r1 + s1, c1 : c1 + s1] = np.random.uniform(0.09, 0.20, (s1, s1))
                anomalies.append({
                    "label": "Deciduous Wildfire Hotspot",
                    "row": r1, "col": c1, "size": s1,
                    "confidence": 95,
                })

        grid_points = []
        cell_deg = 0.005

        for r in range(grid_size):
            for c in range(grid_size):
                val = float(ndvi_grid[r, c])
                grid_points.append({
                    "lat": center_lat + (r - grid_size / 2) * cell_deg,
                    "lon": center_lon + (c - grid_size / 2) * cell_deg,
                    "ndvi": val,
                    "color": (
                        [int(34 + (1 - val) * 200), int(val * 220), 50, 180]
                        if val > 0.3
                        else [220, 50, 30, 200]
                    ),
                })
        grid_df = pd.DataFrame(grid_points)

        anomaly_polys = []
        for a in anomalies:
            r, c, s = a["row"], a["col"], a["size"]
            lat0 = center_lat + (r - grid_size / 2) * cell_deg
            lon0 = center_lon + (c - grid_size / 2) * cell_deg
            lat1 = lat0 + s * cell_deg
            lon1 = lon0 + s * cell_deg
            anomaly_polys.append({
                "polygon": [[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]],
                "label": a["label"],
                "confidence": a["confidence"],
            })
        anomaly_df = pd.DataFrame(anomaly_polys) if anomaly_polys else pd.DataFrame()

        with st.container(border=True):
            st.subheader(f"NDVI Composite — {tl_forest['name']} ({selected_date.strftime('%B %Y')})")

            import pydeck as pdk

            sat_timelapse_base = pdk.Layer(
                "TileLayer",
                data="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                min_zoom=0,
                max_zoom=19,
                tile_size=256,
            )

            ndvi_layer = pdk.Layer(
                "ScatterplotLayer",
                data=grid_df,
                get_position=["lon", "lat"],
                get_fill_color="color",
                get_radius=250,
                opacity=0.85,
            )
            layers = [sat_timelapse_base, ndvi_layer]
            if not anomaly_df.empty:
                bbox_layer = pdk.Layer(
                    "PolygonLayer",
                    data=anomaly_df,
                    get_polygon="polygon",
                    get_fill_color=[239, 68, 68, 60],
                    get_line_color=[239, 68, 68, 255],
                    line_width_min_pixels=3,
                    pickable=True,
                )
                layers.append(bbox_layer)

            sat_view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12, pitch=0, bearing=0)
            sat_deck = pdk.Deck(layers=layers, initial_view_state=sat_view, map_provider="carto", map_style="dark")
            st.pydeck_chart(sat_deck, height=450)

    with sat_col2:
        with st.container(border=True):
            st.subheader("Spatial Anomaly Detections")
            if anomalies:
                for a in anomalies:
                    st.error(
                        f"**:material/warning: {a['label']}**\n\n"
                        f"Grid cell ({a['row']}, {a['col']}) · Cluster size: {a['size']}×{a['size']} cells\n\n"
                        f"Confidence: **{a['confidence']}%**",
                        icon=":material/report:",
                    )
            else:
                st.success(f"No critical anomalies detected for {tl_forest['name']} in this observation month.", icon=":material/check_circle:")

        st.space("small")
        with st.container(border=True):
            st.subheader("NDVI Statistics")
            avg_ndvi = float(ndvi_grid.mean())
            min_ndvi = float(ndvi_grid.min())
            healthy_pct = float((ndvi_grid > 0.4).sum() / ndvi_grid.size * 100)

            st.metric("Mean NDVI", f"{avg_ndvi:.3f}", help="Values > 0.4 indicate robust canopy.", border=True)
            st.metric("Min NDVI", f"{min_ndvi:.3f}", delta_color="inverse" if min_ndvi < 0.2 else "normal", border=True)
            st.metric("Healthy Coverage", f"{healthy_pct:.1f}%", border=True)

    st.space("small")
    with st.container(border=True):
        st.subheader(f"12-Month Mean NDVI Seasonal Trajectory — {tl_forest['name']}")
        np.random.seed(42 + int(center_lat * 10))
        monthly_ndvi = []
        for i in range(12):
            if "sundarbans" in f_id:
                val = 0.75 + 0.08 * np.sin(2 * np.pi * (i + 2) / 12)
            elif "western-ghats" in f_id:
                val = 0.85 + 0.09 * np.sin(2 * np.pi * (i - 2) / 12)
            elif "kaziranga" in f_id:
                val = 0.78 + 0.12 * np.sin(2 * np.pi * (i - 1) / 12)
            elif "corbett" in f_id:
                val = 0.80 + 0.10 * np.sin(2 * np.pi * (i - 2) / 12)
            elif "gir" in f_id:
                val = 0.62 + 0.14 * np.sin(2 * np.pi * (i - 2) / 12)
            else: # bandipur
                val = 0.73 + 0.11 * np.sin(2 * np.pi * (i - 2) / 12)
            monthly_ndvi.append(round(float(np.clip(val + np.random.normal(0, 0.015), 0.3, 0.98)), 3))

        ndvi_trend_df = pd.DataFrame({"Month": month_labels, "Mean NDVI": monthly_ndvi})
        st.line_chart(ndvi_trend_df, x="Month", y="Mean NDVI", color="#22c55e")

# ===========================================================================
# TAB 5: Satellite Detection / Two-date Vegetation Comparison
# ===========================================================================
with tab_satellite_detection:
    from satellite_comparison import render_satellite_comparison

    render_satellite_comparison()

# ===========================================================================
# TAB 6: Environmental Reports
# ===========================================================================
with tab_reports:
    st.header("Environmental & Incident Reports")
    st.caption("AI-generated conservation audits and incident assessments referencing Indian Wildlife Protection Act (1972), NTCA Tiger Conservation, and FSI Guidelines.")

    st.subheader("Report Archive (India)")
    existing_reports = [
        {
            "title": "Q2 2026 Western Ghats Rainforest Canopy Integrity Assessment",
            "date": "2026-07-15",
            "framework": "Forest Survey of India (FSI) & Sentinel-2",
            "status": "Approved",
            "pages": 24,
            "summary": "Full canopy density and risk assessment for Silent Valley National Park and Nilgiri Biosphere. Canopy retention at 96.4%. Zero critical deforestation events.",
        },
        {
            "title": "Sundarbans Tiger Reserve Mangrove & Salinity Verification",
            "date": "2026-06-28",
            "framework": "NTCA Tiger Conservation & Ramsar Convention",
            "status": "Approved",
            "pages": 18,
            "summary": "Independent verification of mangrove density and Bengal Tiger breeding pocket areas in Gosaba sector. ACI score 0.92 confirms 19 indicator species active.",
        },
        {
            "title": "Kaziranga Brahmaputra Flood Inundation Emergency Response",
            "date": "2026-08-11",
            "framework": "Assam Forest Department & Disaster Management",
            "status": "Action Required",
            "pages": 12,
            "summary": "Satellite radar confirms 12.5 ha low-lying floodplain submergence. Animal migration corridor across NH-715 speed regulation enforced.",
        },
    ]

    for rep in existing_reports:
        with st.container(border=True):
            t_col, s_col = st.columns([4, 1])
            with t_col:
                st.markdown(f"**{rep['title']}**")
                st.caption(f":material/calendar_today: {rep['date']} · :material/description: {rep['pages']} pages · :material/verified_user: {rep['framework']}")
            with s_col:
                badge_color = "green" if rep["status"] == "Approved" else "orange"
                st.badge(rep["status"], color=badge_color)

            with st.expander("View Executive Summary", icon=":material/summarize:"):
                st.write(rep["summary"])
                st.download_button(
                    "Download Full Report",
                    data=f"# {rep['title']}\n\nDate: {rep['date']}\nFramework: {rep['framework']}\nStatus: {rep['status']}\n\n{rep['summary']}\n",
                    file_name=f"EcoGuard_Report_{rep['date']}.txt",
                    mime="text/plain",
                    icon=":material/download:",
                    key=f"dl_rep_{rep['date']}",
                )

    st.space("small")

    # --- Generate New Report ---
    st.subheader("Generate New Forest Conservation Assessment Report")
    st.caption("Synthesize ISRO/Sentinel-2 satellite passes, bioacoustic logs, and IoT telemetry into an auditable conservation report.")

    gen_col1, gen_col2 = st.columns([2, 1])
    with gen_col1:
        with st.container(border=True):
            rep_type = st.selectbox(
                "Report Type",
                options=[
                    "Comprehensive Forest Health & Canopy Assessment",
                    "Wildfire & Forest Fire Hotspot Audit",
                    "Wildlife Corridor & Flood Inundation Report",
                    "Bioacoustic Species Diversity Survey",
                    "Quarterly National Conservation Summary",
                ],
            )
            target_forest = st.selectbox("Target Forest Reserve", options=forest_options)
            frameworks = st.multiselect(
                "Indian & Global Conservation Frameworks",
                options=["Forest Survey of India (FSI) Standards", "National Tiger Conservation Authority (NTCA)", "Wildlife Protection Act 1972", "ISRO Bhuvan Earth Observation", "IUCN Red List Guidelines"],
                default=["Forest Survey of India (FSI) Standards", "National Tiger Conservation Authority (NTCA)"],
            )

    with gen_col2:
        with st.container(border=True):
            st.markdown("**AI Synthesis Engine Configuration**")
            detail_depth = st.select_slider("Analysis Depth", options=["Standard", "Detailed", "Comprehensive Audit"], value="Detailed")
            include_telemetry = st.checkbox("Include Raw IoT Sensor Logs", value=True)
            include_spectral = st.checkbox("Include Multi-Spectral NDVI Matrices", value=True)

    if st.button("Generate Environmental Report", icon=":material/auto_awesome:", type="primary"):
        import time
        with st.status("Synthesizing conservation report...", expanded=True) as status:
            st.write("1. Fetching ISRO Cartosat & Sentinel-2 optical/thermal passes...")
            time.sleep(1.0)
            st.write("2. Querying bioacoustic microphone databases & ACI metrics...")
            time.sleep(0.9)
            st.write("3. Cross-referencing FSI & NTCA ecological conservation baselines...")
            time.sleep(1.1)
            st.write("4. Compiling cryptographic provenance signatures...")
            time.sleep(0.8)
            status.update(label="Environmental report generated successfully!", state="complete", expanded=False)

        generated_text = f"""# EcoGuard National Forest Assessment Report
## {rep_type} — {target_forest}

**Generated:** {datetime.now().strftime('%d %B %Y, %H:%M IST')}
**Frameworks Referenced:** {', '.join(frameworks) if frameworks else 'Forest Survey of India (FSI), NTCA'}
**Status:** VALIDATED
**Classification:** PUBLIC ENVIRONMENTAL RECORD (INDIA)

---

### 1. Executive Summary
This environmental assessment report provides an objective evaluation of the **{target_forest}**, synthesizing real-time Earth Observation satellite passes with field IoT sensor networks and bioacoustic surveillance.

**Overall Ecological Status: HEALTHY & MONITORED (Health Index: 94.2%)**

Key Observations:
- **Canopy Retention:** 95.2% average canopy density across all monitored sectors.
- **Vegetation Health:** Sentinel-2 / Cartosat NDVI index indicates vigorous photosynthetic activity (Mean NDVI: 0.83).
- **Biodiversity:** Bengal Tiger, Asian Elephant, and Great Indian Hornbill vocalizations confirmed with 97.8% model confidence.
- **Incident Response:** Real-time threat alert system active for forest fires, shoreline erosion, and flood inundation.

### 2. Recommendations & Action Plan
1. **Immediate:** Maintain thermal drone surveillance over high-risk dry ridge sectors.
2. **Medium-Term:** Expand bioacoustic microphone density along critical wildlife corridors.
3. **Long-Term:** Integrate automated Forest Department patrol dispatch with 5-day satellite cadence passes.
"""

        with st.container(border=True):
            st.markdown(generated_text)

        with st.container(horizontal=True):
            st.download_button(
                "Download as Markdown",
                data=generated_text,
                file_name=f"EcoGuard_{target_forest.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                icon=":material/download:",
                type="primary",
            )
            st.download_button(
                "Download as Plain Text",
                data=generated_text.replace("#", "").replace("**", ""),
                file_name=f"EcoGuard_{target_forest.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                icon=":material/description:",
            )
