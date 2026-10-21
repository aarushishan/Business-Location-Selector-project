"""
Configuration module containing project paths, default business weight profiles,
default budgets, geographic coordinates, and UI color palettes.
"""
from pathlib import Path
from typing import Dict, Any

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# File paths
DATASET_PATH = DATA_DIR / "business_location_synthetic_dataset.csv"
GEOJSON_COORDS_PATH = DATA_DIR / "bhopal_coordinates.json"

# Supported business categories
BUSINESS_TYPES = ["Cafe", "Pharmacy", "Grocery Store", "Clothing Shop"]

# Default business weights (must sum to 1.00)
# Formula: S = w_D*D + w_A*A + w_F*F + w_P*P + w_C*C + w_R*R
DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Cafe": {
        "demand": 0.30,
        "accessibility": 0.20,
        "footfall": 0.15,
        "mobility": 0.15,
        "competition": 0.10,
        "rent": 0.10,
    },
    "Pharmacy": {
        "demand": 0.25,
        "accessibility": 0.25,
        "footfall": 0.10,
        "mobility": 0.15,
        "competition": 0.10,
        "rent": 0.15,
    },
    "Grocery Store": {
        "demand": 0.25,
        "accessibility": 0.15,
        "footfall": 0.20,
        "mobility": 0.15,
        "competition": 0.10,
        "rent": 0.15,
    },
    "Clothing Shop": {
        "demand": 0.25,
        "accessibility": 0.20,
        "footfall": 0.20,
        "mobility": 0.15,
        "competition": 0.10,
        "rent": 0.10,
    },
}

# Default typical budget ceilings (INR per month)
DEFAULT_BUDGETS: Dict[str, int] = {
    "Cafe": 35000,
    "Pharmacy": 30000,
    "Grocery Store": 28000,
    "Clothing Shop": 42000,
}

# Cluster semantic profile mapping
CLUSTER_ARCHETYPES: Dict[int, Dict[str, str]] = {
    0: {
        "name": "Affordable High-Opportunity Hub",
        "badge_color": "#10b981", # Emerald
        "description": "High footfall & demographic density with competitive rents; high demand efficiency."
    },
    1: {
        "name": "Balanced Commercial Core",
        "badge_color": "#3b82f6", # Blue
        "description": "Established high-traffic zone with solid mobility, moderate rents, and healthy demand."
    },
    2: {
        "name": "Prime High-Cost Corridor",
        "badge_color": "#8b5cf6", # Purple
        "description": "Premium central district with maximum brand exposure, but high rent requiring higher margins."
    },
    3: {
        "name": "Suburban / Developing Market",
        "badge_color": "#f59e0b", # Amber
        "description": "Lower rent and low competitor saturation, best suited for destination stores or lean setups."
    }
}

# City centroid coordinates for Bhopal (Madhya Pradesh, India)
BHOPAL_CENTER = [23.2599, 77.4126]
DEFAULT_MAP_ZOOM = 12

# Color scheme for potential tiers
POTENTIAL_COLORS = {
    "Good": "#10b981",      # Emerald Green
    "Moderate": "#3b82f6",  # Vibrant Blue
    "Low": "#ef4444"        # Crimson Red
}
