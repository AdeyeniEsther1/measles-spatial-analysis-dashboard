"""
src/config.py
Central configuration for filepaths, coordinate reference systems,
epidemiological thresholds, and professional visualization palettes.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
BOUNDARIES_DIR = DATA_DIR / "boundaries"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Coordinate Reference Systems
CRS_WGS84 = "EPSG:4326"        # Geographic lat/lon for web maps & Plotly
CRS_BNG = "EPSG:27700"          # British National Grid (metric) for planar calculations

# Public Health Thresholds & Constants
WHO_HERD_IMMUNITY_THRESHOLD_PCT = 95.0
DEFAULT_SIGNIFICANCE_ALPHA = 0.05
RANDOM_SEED = 42

# LISA Cluster Categories & Colors (Standard GeoDa / PySAL conventions)
LISA_COLORS = {
    "High-High (Hotspot)": "#D73027",      # Vibrant Red
    "Low-Low (Coldspot)": "#4575B4",       # Solid Blue
    "High-Low (Outlier)": "#FDAE61",       # Light Orange / Coral
    "Low-High (Buffer)": "#ABD9E9",        # Light Blue / Cyan
    "Not Significant": "#E4E7EB"           # Neutral Slate Grey
}

# Getis-Ord Gi* Confidence Bin Classifications & Colors
GI_STAR_COLORS = {
    "Hot Spot (99% Conf.)": "#B2182B",
    "Hot Spot (95% Conf.)": "#EF8A62",
    "Hot Spot (90% Conf.)": "#FDDBC7",
    "Not Significant": "#F0F2F5",
    "Cold Spot (90% Conf.)": "#D1E5F0",
    "Cold Spot (95% Conf.)": "#67A9CF",
    "Cold Spot (99% Conf.)": "#2166AC"
}

# Continuous Choropleth Color Scales
INCIDENCE_COLORSCALE = "Reds"
VIRIDIS_COLORSCALE = "Viridis"
VACCINE_COLORSCALE = "RdYlGn"   # Red (low coverage) to Green (high coverage)
