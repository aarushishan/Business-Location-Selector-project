"""
Unit tests for geospatial utilities module and OpenStreetMap tile configuration.
"""
import pytest
import pandas as pd
import folium

from src.config import BHOPAL_CENTER, DEFAULT_MAP_ZOOM
from src.data_loader import load_geodata
from src.geo_utils import get_locality_coordinates, build_candidate_map

def test_get_locality_coordinates():
    geo_dict = load_geodata()
    coords = get_locality_coordinates("New Market", geo_dict)
    assert coords is not None
    assert "lat" in coords and "lon" in coords
    assert 23.0 <= coords["lat"] <= 24.0
    assert 77.0 <= coords["lon"] <= 78.0

    # Test partial / case-insensitive match
    coords_partial = get_locality_coordinates("new market central", geo_dict)
    assert coords_partial is not None

def test_build_candidate_map_tiles_and_attribution():
    data = {
        "location_name": ["New Market", "MP Nagar", "Kolar Road"],
        "rank": [1, 2, 3],
        "suitability_score": [82.5, 68.0, 48.0],
        "monthly_rent": [35000, 42000, 22000],
        "daily_footfall": [2100, 1800, 1100],
        "predicted_demand": [180.0, 140.0, 95.0],
        "competitor_count": [4, 7, 2],
        "cluster_name": ["Affordable High-Opportunity Hub", "Balanced Commercial Core", "Suburban / Developing Market"],
        "business_type": ["Cafe", "Cafe", "Cafe"]
    }
    df = pd.DataFrame(data)
    fmap = build_candidate_map(df, top_n=3, highlight_rank_1=True)

    assert isinstance(fmap, folium.Map)
    assert fmap.location == BHOPAL_CENTER
    assert fmap.location == [23.2599, 77.4126]
    assert fmap.options["zoom"] == DEFAULT_MAP_ZOOM

    # Render HTML and verify tile URL, attribution, and absence of broken Carto tiles
    html = fmap.get_root().render()
    assert "https://tile.openstreetmap.org/{z}/{x}/{y}.png" in html
    assert "OpenStreetMap" in html
    assert "contributors" in html
    assert "cartodb" not in html.lower()
    assert "carto.com" not in html.lower()

def test_map_markers_and_popups():
    data = {
        "location_name": ["New Market", "MP Nagar", "Kolar Road"],
        "rank": [1, 2, 3],
        "suitability_score": [82.5, 68.0, 48.0],  # 1 Top tier (green), 1 Moderate (blue), 1 Caution (red)
        "monthly_rent": [35000, 42000, 22000],
        "daily_footfall": [2100, 1800, 1100],
        "predicted_demand": [180.0, 140.0, 95.0],
        "competitor_count": [4, 7, 2],
        "cluster_name": ["Affordable High-Opportunity Hub", "Balanced Commercial Core", "Suburban / Developing Market"],
        "business_type": ["Cafe", "Cafe", "Cafe"]
    }
    df = pd.DataFrame(data)
    fmap = build_candidate_map(df, top_n=3, highlight_rank_1=True)
    html = fmap.get_root().render()

    # Emerald Green for Score >= 75
    assert "#10b981" in html
    # Blue for Score >= 55
    assert "#3b82f6" in html
    # Red for Score < 55
    assert "#ef4444" in html

    # Check popup contents
    assert "New Market" in html
    assert "MP Nagar" in html
    assert "Kolar Road" in html
    assert "TOP TIER" in html
    assert "MODERATE" in html
    assert "CAUTION" in html
