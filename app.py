import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFilter
import os
import io

# 1. Page Configuration
st.set_page_config(
    page_title="ECO-MRV AGENT Â· Environmental Intelligence & Forest Monitoring",
    page_icon="ðŸŒ¿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ðŸŒ¿ GREENERY 3D THEME â€” Full immersive forest canopy UI
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<style>
/* â”€â”€ Import premium font â”€â”€ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* â”€â”€ Root variables â”€â”€ */
:root {
    --forest-darkest: #050d08;
    --forest-dark: #0a1a0f;
    --forest-mid: #122a1a;
    --forest-light: #1a3d26;
    --forest-canopy: #1e4d30;
    --emerald: #22c55e;
    --emerald-glow: #34d369;
    --emerald-soft: #4ade80;
    --lime-accent: #a3e635;
    --moss: #6b8f5e;
    --mint: #b8f5d0;
    --text-primary: #e2f5e9;
    --text-secondary: #98c4a8;
    --text-muted: #5a8a6a;
    --glass-bg: rgba(18, 42, 26, 0.55);
    --glass-border: rgba(34, 197, 94, 0.18);
    --glass-shine: rgba(74, 222, 128, 0.06);
    --shadow-green: rgba(34, 197, 94, 0.12);
    --shadow-dark: rgba(0, 0, 0, 0.35);
}

/* â”€â”€ Global body & app â”€â”€ */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: var(--forest-darkest) !important;
}

/* â”€â”€ Animated forest canopy gradient background â”€â”€ */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 120% 60% at 20% 10%, rgba(34, 197, 94, 0.07) 0%, transparent 55%),
        radial-gradient(ellipse 100% 50% at 80% 90%, rgba(30, 77, 48, 0.12) 0%, transparent 50%),
        radial-gradient(ellipse 80% 80% at 50% 50%, rgba(10, 26, 15, 1) 0%, rgba(5, 13, 8, 1) 100%) !important;
}

/* â”€â”€ Floating leaf particles (pure CSS) â”€â”€ */
[data-testid="stAppViewContainer"]::before {
    content: 'ðŸŒ¿';
    position: fixed;
    top: -30px;
    left: 10%;
    font-size: 22px;
    opacity: 0.12;
    animation: leafFloat1 18s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}
[data-testid="stAppViewContainer"]::after {
    content: 'ðŸƒ';
    position: fixed;
    top: -30px;
    right: 20%;
    font-size: 18px;
    opacity: 0.10;
    animation: leafFloat2 22s ease-in-out infinite 3s;
    pointer-events: none;
    z-index: 0;
}

@keyframes leafFloat1 {
    0%   { transform: translate(0, 0) rotate(0deg); opacity: 0; }
    10%  { opacity: 0.15; }
    50%  { transform: translate(120px, 50vh) rotate(180deg); opacity: 0.10; }
    90%  { opacity: 0.05; }
    100% { transform: translate(60px, 105vh) rotate(360deg); opacity: 0; }
}
@keyframes leafFloat2 {
    0%   { transform: translate(0, 0) rotate(0deg); opacity: 0; }
    10%  { opacity: 0.12; }
    50%  { transform: translate(-80px, 55vh) rotate(-150deg); opacity: 0.08; }
    90%  { opacity: 0.03; }
    100% { transform: translate(-40px, 108vh) rotate(-320deg); opacity: 0; }
}

/* â”€â”€ Sidebar â€” deep forest panel with glass â”€â”€ */
[data-testid="stSidebar"] {
    background: linear-gradient(
        175deg,
        rgba(18, 42, 26, 0.94) 0%,
        rgba(10, 26, 15, 0.98) 100%
    ) !important;
    backdrop-filter: blur(20px) saturate(1.5) !important;
    -webkit-backdrop-filter: blur(20px) saturate(1.5) !important;
    border-right: 1px solid var(--glass-border) !important;
    box-shadow:
        4px 0 30px rgba(0, 0, 0, 0.3),
        1px 0 0 rgba(34, 197, 94, 0.08) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdown"] {
    color: var(--text-secondary) !important;
}

/* â”€â”€ 3D Glass containers â€” main card style â”€â”€ */
[data-testid="stVerticalBlock"] > div > div[data-testid="stContainer"],
[data-testid="stExpander"],
div[data-testid="stMetric"] {
    perspective: 1200px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(16px) saturate(1.4) !important;
    -webkit-backdrop-filter: blur(16px) saturate(1.4) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    box-shadow:
        0 8px 32px var(--shadow-dark),
        0 2px 8px var(--shadow-green),
        inset 0 1px 0 var(--glass-shine) !important;
    transition: all 0.3s cubic-bezier(0.23, 1, 0.32, 1) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(34, 197, 94, 0.35) !important;
    box-shadow:
        0 14px 40px var(--shadow-dark),
        0 4px 16px var(--shadow-green),
        0 0 20px rgba(34, 197, 94, 0.08),
        inset 0 1px 0 rgba(74, 222, 128, 0.12) !important;
    transform: translateY(-2px) !important;
}

/* â”€â”€ Metric cards â”€â”€ */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(18, 42, 26, 0.7) 0%, rgba(10, 26, 15, 0.85) 100%) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    box-shadow: inset 0 1px 0 var(--glass-shine), 0 4px 16px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.25s ease !important;
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(34, 197, 94, 0.4) !important;
    box-shadow: inset 0 1px 0 rgba(74, 222, 128, 0.15), 0 6px 20px rgba(34, 197, 94, 0.15) !important;
}
div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    text-shadow: 0 0 12px rgba(34, 197, 94, 0.3) !important;
}

/* â”€â”€ Buttons â€” glowing emerald â”€â”€ */
.stButton > button {
    background: linear-gradient(135deg, #166534 0%, #15803d 100%) !important;
    color: #e2f5e9 !important;
    border: 1px solid rgba(74, 222, 128, 0.3) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.3px !important;
    padding: 8px 18px !important;
    box-shadow: 0 4px 14px rgba(22, 101, 52, 0.35), inset 0 1px 0 rgba(74, 222, 128, 0.2) !important;
    transition: all 0.25s cubic-bezier(0.23, 1, 0.32, 1) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #15803d 0%, #22c55e 100%) !important;
    color: #ffffff !important;
    border-color: #4ade80 !important;
    box-shadow: 0 6px 20px rgba(34, 197, 94, 0.45), 0 0 15px rgba(34, 197, 94, 0.2) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #15803d 0%, #22c55e 100%) !important;
    box-shadow: 0 4px 18px rgba(34, 197, 94, 0.45) !important;
}

/* â”€â”€ Headers & accents â”€â”€ */
h1, h2, h3 {
    color: var(--text-primary) !important;
    letter-spacing: -0.3px !important;
}
h1 {
    text-shadow: 0 0 20px rgba(34, 197, 94, 0.2) !important;
}
</style>
""", unsafe_allow_html=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. Model Warmup - Skipped at startup; models load lazily on first use
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# (Heavy ML models like OWL-ViT and CLAP are loaded on demand to avoid
#  blocking the UI at startup. No spinner needed here.)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. Sidebar: ECO-MRV AGENT Branding & Environmental Controls
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Only the selected page executes; wildlife pages do not run dashboard requests.
page = st.navigation([
    st.Page("forest_dashboard.py", title="Forest Dashboard", icon=":material/forest:", default=True),
    st.Page("forest_assistant.py", title="AI Forest Assistant", icon=":material/auto_awesome:", url_path="ai-forest-assistant"),
    st.Page("bioacoustics.py", title="Bioacoustics", icon=":material/graphic_eq:", url_path="bioacoustics"),
    st.Page("live_camera.py", title="Live Camera & Video", icon=":material/videocam:", url_path="live-camera"),
    st.Page("satellite_ndvi.py", title="Satellite / NDVI", icon=":material/satellite_alt:", url_path="satellite"),
    st.Page("wildlife_reports.py", title="Reports", icon=":material/assignment:", url_path="reports"),
])
page.run()


