"""
Model Trainer module for supervised regression:
Trains, benchmarks, evaluates, and serializes Linear Regression, Decision Tree,
and Random Forest Regressors for commercial demand prediction.
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import MODELS_DIR
from src.data_loader import load_raw_data
from src.feature_engineering import FeaturePipeline

def train_and_evaluate_models(
    df: Optional[pd.DataFrame] = None
) -> Tuple[Dict[str, Any], FeaturePipeline, Dict[str, Any]]:
    """
    Train Linear Regression, Decision Tree, and Random Forest models on candidate data.

    Returns:
        models_dict, fitted feature_pipeline, metrics_summary
    """
    if df is None:
        df = load_raw_data()

    X_raw = df.drop(columns=["target_demand"])
    y = df["target_demand"].values

    # Train-test split (80/20) with random seed for reproducibility
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42
    )

    # Fit feature transformation pipeline strictly on training data
    pipeline = FeaturePipeline()
    X_train = pipeline.fit_transform(X_train_raw)
    X_test = pipeline.transform(X_test_raw)

    # Initialize models
    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_split=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=10, min_samples_split=6, random_state=42, n_jobs=-1)
    }

    trained_models = {}
    metrics_summary = {}

    for name, model in candidate_models.items():
        # Fit model
        model.fit(X_train, y_train)
        trained_models[name] = model

        # In-sample & Out-of-sample predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 5-fold cross validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

        metrics_summary[name] = {
            "train_r2": float(r2_score(y_train, y_train_pred)),
            "test_r2": float(r2_score(y_test, y_test_pred)),
            "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
            "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
        }

    # Extract feature importances from Random Forest
    rf_model = trained_models["Random Forest"]
    importances = rf_model.feature_importances_
    feat_importance_dict = {
        name: float(imp)
        for name, imp in sorted(zip(pipeline.feature_names, importances), key=lambda x: x[1], reverse=True)
    }

    summary = {
        "metrics": metrics_summary,
        "feature_importances": feat_importance_dict,
        "total_train_samples": len(y_train),
        "total_test_samples": len(y_test),
        "feature_count": len(pipeline.feature_names)
    }

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "feature_pipeline.joblib")
    joblib.dump(trained_models["Random Forest"], MODELS_DIR / "rf_regressor.joblib")
    joblib.dump(trained_models["Decision Tree"], MODELS_DIR / "dt_regressor.joblib")
    joblib.dump(trained_models["Linear Regression"], MODELS_DIR / "linear_regressor.joblib")

    with open(MODELS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return trained_models, pipeline, summary

def load_trained_models() -> Tuple[Dict[str, Any], FeaturePipeline, Dict[str, Any]]:
    """
    Load pre-trained models from disk. If not found, train them on demand.
    """
    rf_path = MODELS_DIR / "rf_regressor.joblib"
    pipe_path = MODELS_DIR / "feature_pipeline.joblib"
    metrics_path = MODELS_DIR / "model_metrics.json"

    if not (rf_path.exists() and pipe_path.exists() and metrics_path.exists()):
        return train_and_evaluate_models()

    models = {
        "Random Forest": joblib.load(rf_path),
        "Decision Tree": joblib.load(MODELS_DIR / "dt_regressor.joblib"),
        "Linear Regression": joblib.load(MODELS_DIR / "linear_regressor.joblib")
    }
    pipeline = joblib.load(pipe_path)
    with open(metrics_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    return models, pipeline, summary

if __name__ == "__main__":
    print("Training and benchmarking models...")
    models, pipe, summary = train_and_evaluate_models()
    print("Model Training Complete! Summary:")
    print(json.dumps(summary["metrics"], indent=2))
