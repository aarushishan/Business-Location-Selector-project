"""
Clustering module for candidate location market segmentation using K-Means,
Elbow method (Inertia), and Silhouette Score validation.
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, CLUSTER_ARCHETYPES
from src.data_loader import load_raw_data
from src.feature_engineering import prepare_cluster_features

def evaluate_and_fit_clusters(
    df: Optional[pd.DataFrame] = None,
    n_clusters: int = 4
) -> Tuple[KMeans, StandardScaler, List[str], Dict[str, Any], pd.DataFrame]:
    """
    Evaluate K-Means across k in [2, 7] and fit the optimal cluster model.

    Args:
        df: Input DataFrame.
        n_clusters: Number of clusters to fit (default: 4).

    Returns:
        fitted_kmeans, fitted_scaler, cluster_features, evaluation_metrics, df_with_clusters
    """
    if df is None:
        df = load_raw_data()

    X_scaled, scaler, cluster_cols = prepare_cluster_features(df)

    # Evaluate k from 2 to 7
    k_range = list(range(2, 8))
    inertias = []
    silhouette_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        silhouette_scores.append(float(silhouette_score(X_scaled, labels)))

    # Fit selected K-Means model
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    cluster_labels = kmeans_model.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered["cluster_id"] = cluster_labels
    df_clustered["cluster_name"] = [
        CLUSTER_ARCHETYPES.get(cid, {}).get("name", f"Cluster {cid}")
        for cid in cluster_labels
    ]

    # Cluster profiles (averages per cluster)
    profile_summary = {}
    for cid in range(n_clusters):
        subset = df_clustered[df_clustered["cluster_id"] == cid]
        profile_summary[cid] = {
            "name": CLUSTER_ARCHETYPES.get(cid, {}).get("name", f"Cluster {cid}"),
            "count": int(len(subset)),
            "pct": float(round(len(subset) / len(df_clustered) * 100, 1)),
            "mean_rent": float(round(subset["monthly_rent"].mean(), 1)),
            "mean_footfall": float(round(subset["daily_footfall"].mean(), 1)),
            "mean_competitors": float(round(subset["competitor_count"].mean(), 1)),
            "mean_accessibility": float(round(subset["accessibility_score"].mean(), 1)),
            "mean_demand": float(round(subset["target_demand"].mean(), 1)),
        }

    clustering_metrics = {
        "k_range": k_range,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "selected_k": n_clusters,
        "selected_silhouette": float(silhouette_score(X_scaled, cluster_labels)),
        "cluster_profiles": profile_summary
    }

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans_model, MODELS_DIR / "kmeans_clusterer.joblib")
    joblib.dump(scaler, MODELS_DIR / "kmeans_scaler.joblib")

    with open(MODELS_DIR / "clustering_metrics.json", "w", encoding="utf-8") as f:
        json.dump(clustering_metrics, f, indent=2)

    return kmeans_model, scaler, cluster_cols, clustering_metrics, df_clustered

def load_trained_clusterer() -> Tuple[KMeans, StandardScaler, List[str], Dict[str, Any]]:
    """
    Load pre-trained clustering model from disk or train if missing.
    """
    km_path = MODELS_DIR / "kmeans_clusterer.joblib"
    scaler_path = MODELS_DIR / "kmeans_scaler.joblib"
    metrics_path = MODELS_DIR / "clustering_metrics.json"

    if not (km_path.exists() and scaler_path.exists() and metrics_path.exists()):
        km, scaler, cols, metrics, _ = evaluate_and_fit_clusters()
        return km, scaler, cols, metrics

    km = joblib.load(km_path)
    scaler = joblib.load(scaler_path)
    cluster_cols = [
        "monthly_rent",
        "daily_footfall",
        "competitor_count",
        "accessibility_score",
        "population"
    ]
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return km, scaler, cluster_cols, metrics

if __name__ == "__main__":
    print("Evaluating and fitting K-Means clusters...")
    km, scaler, cols, metrics, df_c = evaluate_and_fit_clusters()
    print("Clustering Complete!")
    print(f"Selected k: {metrics['selected_k']}, Silhouette: {metrics['selected_silhouette']:.3f}")
    print(json.dumps(metrics["cluster_profiles"], indent=2))
