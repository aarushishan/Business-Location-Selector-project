"""
Unit tests for scoring engine module.
"""
import pytest
import pandas as pd
import numpy as np

from src.scoring_engine import (
    normalize_accessibility,
    normalize_footfall,
    normalize_demand,
    normalize_competition,
    compute_rent_affordability,
    calculate_suitability_score,
    rank_candidate_locations
)

def test_normalization_bounds():
    assert normalize_accessibility(1.0) == 0.0
    assert normalize_accessibility(10.0) == 100.0
    assert 0.0 <= normalize_footfall(1500.0) <= 100.0
    assert 0.0 <= normalize_demand(120.0) <= 100.0
    assert normalize_competition(0.0) == 100.0

def test_rent_affordability():
    budget = 40000.0
    # Rent half budget -> 100
    assert compute_rent_affordability(20000.0, budget) == 100.0
    # Rent equal budget -> 40
    assert compute_rent_affordability(40000.0, budget) == 40.0
    # Rent above budget -> < 40
    assert compute_rent_affordability(50000.0, budget) < 40.0

def test_suitability_score_calculation():
    score = calculate_suitability_score(
        demand_sub=90.0,
        accessibility_sub=80.0,
        footfall_sub=85.0,
        mobility_sub=90.0,
        competition_sub=65.0,
        rent_sub=84.0
    )
    assert 70.0 <= score <= 95.0

def test_rank_candidates():
    data = {
        "location_name": ["LocA", "LocB"],
        "business_type": ["Cafe", "Cafe"],
        "monthly_rent": [35000, 25000],
        "population": [25000, 40000],
        "competitor_count": [6, 2],
        "daily_footfall": [1500, 2200],
        "accessibility_score": [8, 9],
        "parking_score": [7, 8],
        "public_transport_score": [8, 9],
        "target_demand": [120, 160]
    }
    df = pd.DataFrame(data)
    ranked = rank_candidate_locations(df, business_type="Cafe", budget=35000)
    assert len(ranked) == 2
    assert ranked.iloc[0]["rank"] == 1
    assert ranked.iloc[0]["suitability_score"] >= ranked.iloc[1]["suitability_score"]
