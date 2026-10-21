"""
Data Loader module for loading, validating, and preprocessing candidate location records.
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from src.config import DATASET_PATH, GEOJSON_COORDS_PATH

REQUIRED_COLUMNS = [
    "location_name",
    "business_type",
    "monthly_rent",
    "population",
    "competitor_count",
    "daily_footfall",
    "accessibility_score",
    "parking_score",
    "public_transport_score",
    "income_level",
    "nearby_facilities",
    "target_demand"
]

def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load candidate location dataset from CSV file and perform initial hygiene checks.

    Args:
        filepath: Optional custom Path to CSV. Defaults to config.DATASET_PATH.

    Returns:
        Cleaned pandas DataFrame.
    """
    path = filepath or DATASET_PATH
    if not path.exists():
        # Fallback to local root file if running from root
        alt_path = Path("business_location_synthetic_dataset.csv")
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    # Validate column presence
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()

    # Numeric hygiene: enforce non-negative values
    num_cols = ["monthly_rent", "population", "competitor_count", "daily_footfall", "target_demand"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            df[c] = df[c].clip(lower=0)

    # Score bounded validation [1, 10]
    score_cols = ["accessibility_score", "parking_score", "public_transport_score"]
    for c in score_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(5).clip(lower=1, upper=10)

    # Ensure mobility score exists (rescaled average of parking and public transport 0-100)
    if "mobility_score" not in df.columns:
        df["mobility_score"] = ((df["parking_score"] + df["public_transport_score"] - 2) / 18.0) * 100.0

    return df

def load_geodata(filepath: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load Bhopal geographic coordinates lookup table from JSON.

    Args:
        filepath: Optional custom Path to JSON. Defaults to config.GEOJSON_COORDS_PATH.

    Returns:
        Dictionary mapping locality name to lat, lon, zone, and description.
    """
    path = filepath or GEOJSON_COORDS_PATH
    if not path.exists():
        alt_path = Path("data/bhopal_coordinates.json")
        if alt_path.exists():
            path = alt_path
        else:
            return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate key summary statistics for the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Summary dictionary with counts, averages, and unique categories.
    """
    return {
        "total_records": len(df),
        "unique_locations": int(df["location_name"].nunique()),
        "business_types": df["business_type"].value_counts().to_dict(),
        "rent_range": (float(df["monthly_rent"].min()), float(df["monthly_rent"].max())),
        "rent_mean": float(df["monthly_rent"].mean()),
        "footfall_range": (float(df["daily_footfall"].min()), float(df["daily_footfall"].max())),
        "footfall_mean": float(df["daily_footfall"].mean()),
        "demand_range": (float(df["target_demand"].min()), float(df["target_demand"].max())),
        "demand_mean": float(df["target_demand"].mean()),
    }

if __name__ == "__main__":
    df = load_raw_data()
    print("Dataset loaded successfully!")
    print(get_dataset_summary(df))
