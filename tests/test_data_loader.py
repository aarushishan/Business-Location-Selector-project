"""
Unit tests for data loader module.
"""
from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import load_raw_data, load_geodata, get_dataset_summary, REQUIRED_COLUMNS

def test_load_raw_data():
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 500
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    # Ensure scores bounded
    assert (df["accessibility_score"] >= 1).all()
    assert (df["accessibility_score"] <= 10).all()

def test_load_geodata():
    geo = load_geodata()
    assert isinstance(geo, dict)
    assert "New Market" in geo
    assert "lat" in geo["New Market"]
    assert "lon" in geo["New Market"]

def test_dataset_summary():
    df = load_raw_data()
    summary = get_dataset_summary(df)
    assert summary["total_records"] == len(df)
    assert summary["unique_locations"] > 10
    assert summary["rent_mean"] > 10000
