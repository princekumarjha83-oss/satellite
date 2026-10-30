"""Configuration for EcoMRV's project-specific evidence indicators.

These values are prototype defaults, not scientifically validated ecological
thresholds. Keep them in one place so deployments can calibrate them openly.
"""

FUSION_WEIGHTS = {"camera": 0.30, "audio": 0.30, "satellite": 0.40}

# MEHS is intentionally distinct from fusion: it rewards corroboration and
# reduces the indicator when a historical deviation is available.
MEHS_WEIGHTS = {"fusion": 0.75, "agreement": 0.10, "anomaly_risk": 0.15}

AGREEMENT_LEVELS = ((75.0, "high"), (50.0, "moderate"), (0.0, "low"))

# Project/configuration thresholds, not validated deforestation thresholds.
NDVI_CHANGE_THRESHOLDS = {"decrease": -0.05, "increase": 0.05, "stable": 0.02}
