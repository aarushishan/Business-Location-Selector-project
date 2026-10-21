"""
Feature Engineering module for encoding categorical variables, constructing domain ratios,
and scaling feature vectors for supervised regression and unsupervised clustering.
"""
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

NUMERICAL_FEATURES = [
    "monthly_rent",
    "population",
    "competitor_count",
    "daily_footfall",
    "accessibility_score",
    "parking_score",
    "public_transport_score",
    "footfall_per_rent",
    "pop_per_competitor",
    "mobility_index"
]

ORDINAL_FEATURES = ["income_level_num"]
CATEGORICAL_FEATURES = ["business_type", "nearby_facilities"]

INCOME_MAP = {
    "Low": 1.0,
    "Medium": 2.0,
    "High": 3.0
}

def engineer_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct domain-specific financial and accessibility ratio features.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with new engineered feature columns.
    """
    df = df.copy()

    # Ratios
    df["footfall_per_rent"] = df["daily_footfall"] / (df["monthly_rent"] + 1.0)
    df["pop_per_competitor"] = df["population"] / (df["competitor_count"] + 1.0)
    df["mobility_index"] = (
        df["accessibility_score"] + df["parking_score"] + df["public_transport_score"]
    ) / 3.0

    # Ordinal income encoding
    if "income_level" in df.columns:
        df["income_level_num"] = df["income_level"].map(INCOME_MAP).fillna(2.0)
    else:
        df["income_level_num"] = 2.0

    return df

class FeaturePipeline:
    """
    Feature transformation pipeline that handles scaling and one-hot encoding
    without data leakage.
    """
    def __init__(self):
        self.column_transformer: Optional[ColumnTransformer] = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        df_feat = engineer_domain_features(df)

        num_cols = NUMERICAL_FEATURES + ORDINAL_FEATURES
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in df_feat.columns]

        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
            ]
        )

        self.column_transformer.fit(df_feat)
        self.is_fitted = True

        # Extract generated feature names
        num_names = num_cols
        cat_encoder = self.column_transformer.named_transformers_["cat"]
        if hasattr(cat_encoder, "get_feature_names_out"):
            cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
        else:
            cat_names = [f"cat_{i}" for i in range(len(cat_encoder.categories_))]

        self.feature_names = num_names + cat_names
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.column_transformer is None:
            raise RuntimeError("FeaturePipeline must be fitted before transform can be called.")
        df_feat = engineer_domain_features(df)
        return self.column_transformer.transform(df_feat)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

def prepare_cluster_features(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, List[str]]:
    """
    Prepares normalized features specifically for K-Means spatial and economic clustering.

    Features used:
    - monthly_rent
    - daily_footfall
    - competitor_count
    - accessibility_score
    - population

    Returns:
        X_scaled, fitted StandardScaler, feature list
    """
    cluster_cols = [
        "monthly_rent",
        "daily_footfall",
        "competitor_count",
        "accessibility_score",
        "population"
    ]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cluster_cols])
    return X_scaled, scaler, cluster_cols
