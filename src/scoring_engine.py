"""
Scoring Engine module implementing the multi-criteria suitability formula:
S = w_D * D + w_A * A + w_F * F + w_P * P + w_C * C + w_R * R
Calculates component sub-scores, handles dynamic weight sensitivity,
and generates explainable location rankings.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.config import DEFAULT_WEIGHTS, DEFAULT_BUDGETS

def normalize_accessibility(accessibility_score: float) -> float:
    """Normalize 1-10 accessibility score to 0-100."""
    return float(np.clip(((accessibility_score - 1.0) / 9.0) * 100.0, 0.0, 100.0))

def normalize_mobility(parking_score: float, transit_score: float) -> float:
    """Combine parking and public transport scores into a 0-100 mobility score."""
    p_norm = ((parking_score - 1.0) / 9.0) * 100.0
    t_norm = ((transit_score - 1.0) / 9.0) * 100.0
    return float(np.clip((p_norm + t_norm) / 2.0, 0.0, 100.0))

def normalize_footfall(footfall: float, f_min: float = 200.0, f_max: float = 2700.0) -> float:
    """Normalize daily footfall to 0-100."""
    if f_max <= f_min:
        return 50.0
    norm = ((footfall - f_min) / (f_max - f_min)) * 100.0
    return float(np.clip(norm, 0.0, 100.0))

def normalize_demand(demand: float, d_min: float = 40.0, d_max: float = 200.0) -> float:
    """Normalize demand / customer count to 0-100."""
    if d_max <= d_min:
        return 50.0
    norm = ((demand - d_min) / (d_max - d_min)) * 100.0
    return float(np.clip(norm, 0.0, 100.0))

def normalize_competition(competitor_count: float, max_competitors: float = 14.0) -> float:
    """
    Compute competition opportunity score [0-100].
    Fewer competitors receive higher opportunity scores.
    """
    score = 100.0 * (1.0 - (competitor_count / max_competitors))
    return float(np.clip(score, 0.0, 100.0))

def compute_rent_affordability(rent: float, budget: float) -> float:
    """
    Compute piecewise rent affordability score [0-100] based on user budget.

    - Rent <= 0.50 * Budget -> 100 (Maximum affordability)
    - 0.50 * Budget < Rent <= Budget -> Linear decay from 100 to 40
    - Rent > Budget -> Penalized decay from 40 to 0
    """
    if budget <= 0:
        return 50.0

    ratio = rent / budget
    if ratio <= 0.50:
        return 100.0
    elif ratio <= 1.0:
        # Decays from 100 down to 40
        return float(100.0 - 60.0 * ((rent - 0.50 * budget) / (0.50 * budget)))
    else:
        # Decays from 40 down to 0
        penalized = 40.0 - 80.0 * (ratio - 1.0)
        return float(np.clip(penalized, 0.0, 40.0))

def calculate_suitability_score(
    demand_sub: float,
    accessibility_sub: float,
    footfall_sub: float,
    mobility_sub: float,
    competition_sub: float,
    rent_sub: float,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Master Suitability Score calculation:
    S = w_D*D + w_A*A + w_F*F + w_P*P + w_C*C + w_R*R
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS["Cafe"]

    # Normalize weights so they strictly sum to 1.00
    total_w = sum(weights.values())
    if total_w <= 0:
        total_w = 1.0
    w = {k: v / total_w for k, v in weights.items()}

    score = (
        w.get("demand", 0.30) * demand_sub +
        w.get("accessibility", 0.20) * accessibility_sub +
        w.get("footfall", 0.15) * footfall_sub +
        w.get("mobility", 0.15) * mobility_sub +
        w.get("competition", 0.10) * competition_sub +
        w.get("rent", 0.10) * rent_sub
    )
    return float(np.clip(score, 0.0, 100.0))

def rank_candidate_locations(
    df: pd.DataFrame,
    business_type: str = "Cafe",
    budget: Optional[float] = None,
    custom_weights: Optional[Dict[str, float]] = None,
    predicted_demands: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Filter, score, and rank candidate locations for a given business type and budget.

    Returns:
        Ranked DataFrame sorted descending by final suitability score,
        with all sub-scores and trade-off explanations.
    """
    # Filter by business type if column exists
    if "business_type" in df.columns:
        df_sub = df[df["business_type"] == business_type].copy()
        if len(df_sub) == 0:
            df_sub = df.copy()
    else:
        df_sub = df.copy()

    # Determine user budget
    effective_budget = budget if budget is not None else DEFAULT_BUDGETS.get(business_type, 35000)

    # Weights
    weights = custom_weights or DEFAULT_WEIGHTS.get(business_type, DEFAULT_WEIGHTS["Cafe"])

    # Determine demand vector
    if predicted_demands is not None and len(predicted_demands) == len(df_sub):
        demands = predicted_demands
    elif "target_demand" in df_sub.columns:
        demands = df_sub["target_demand"].values
    else:
        demands = np.full(len(df_sub), 100.0)

    f_min = float(df["daily_footfall"].min()) if "daily_footfall" in df.columns else 200.0
    f_max = float(df["daily_footfall"].max()) if "daily_footfall" in df.columns else 2700.0
    d_min = float(df["target_demand"].min()) if "target_demand" in df.columns else 40.0
    d_max = float(df["target_demand"].max()) if "target_demand" in df.columns else 200.0

    # Compute individual sub-scores
    sub_demand = [normalize_demand(d, d_min, d_max) for d in demands]
    sub_access = [normalize_accessibility(a) for a in df_sub["accessibility_score"]]
    sub_footfall = [normalize_footfall(f, f_min, f_max) for f in df_sub["daily_footfall"]]
    sub_mobility = [
        normalize_mobility(p, t)
        for p, t in zip(df_sub["parking_score"], df_sub["public_transport_score"])
    ]
    sub_comp = [normalize_competition(c) for c in df_sub["competitor_count"]]
    sub_rent = [compute_rent_affordability(r, effective_budget) for r in df_sub["monthly_rent"]]

    # Final score
    final_scores = [
        calculate_suitability_score(d, a, f, m, c, r, weights)
        for d, a, f, m, c, r in zip(
            sub_demand, sub_access, sub_footfall, sub_mobility, sub_comp, sub_rent
        )
    ]

    df_sub["predicted_demand"] = np.round(demands, 1)
    df_sub["sub_demand"] = np.round(sub_demand, 1)
    df_sub["sub_access"] = np.round(sub_access, 1)
    df_sub["sub_footfall"] = np.round(sub_footfall, 1)
    df_sub["sub_mobility"] = np.round(sub_mobility, 1)
    df_sub["sub_competition"] = np.round(sub_comp, 1)
    df_sub["sub_rent_affordability"] = np.round(sub_rent, 1)
    df_sub["suitability_score"] = np.round(final_scores, 2)

    # Sort descending
    df_ranked = df_sub.sort_values(by="suitability_score", ascending=False).reset_index(drop=True)
    df_ranked["rank"] = df_ranked.index + 1

    # Generate natural language explanation for strengths & trade-offs
    explanations = []
    for _, row in df_ranked.iterrows():
        strengths = []
        tradeoffs = []

        if row["sub_demand"] >= 75:
            strengths.append(f"High projected demand ({row['predicted_demand']:.0f}/day)")
        if row["sub_footfall"] >= 70:
            strengths.append(f"Heavy pedestrian traffic ({row['daily_footfall']} footfall)")
        if row["sub_rent_affordability"] >= 80:
            strengths.append(f"Comfortably within budget (₹{row['monthly_rent']:,})")
        if row["sub_mobility"] >= 70:
            strengths.append("Excellent parking & transit connectivity")

        if row["monthly_rent"] > effective_budget:
            tradeoffs.append(f"Exceeds monthly budget by ₹{row['monthly_rent'] - effective_budget:,}")
        if row["competitor_count"] >= 8:
            tradeoffs.append(f"Dense competition ({row['competitor_count']} competitors nearby)")
        if row["sub_access"] < 45:
            tradeoffs.append("Sub-optimal road accessibility / vehicular access")

        exp_str = ""
        if strengths:
            exp_str += "Strengths: " + "; ".join(strengths) + ". "
        if tradeoffs:
            exp_str += "Trade-offs: " + "; ".join(tradeoffs) + "."
        if not exp_str:
            exp_str = "Balanced mid-tier location across commercial indicators."

        explanations.append(exp_str)

    df_ranked["trade_off_explanation"] = explanations
    return df_ranked
