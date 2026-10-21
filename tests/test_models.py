"""
Unit tests for model loading, inference and clustering.
"""
import pytest
import numpy as np
import pandas as pd

from src.data_loader import load_raw_data
from src.model_trainer import load_trained_models
from src.clustering import load_trained_clusterer

def test_models_inference():
    models, pipe, summary = load_trained_models()
    assert "Random Forest" in models
    assert "Decision Tree" in models
    assert "Linear Regression" in models
    assert pipe.is_fitted

    df = load_raw_data().head(5)
    X = pipe.transform(df)
    preds = models["Random Forest"].predict(X)
    assert len(preds) == 5
    assert (preds > 0).all()

def test_clustering_prediction():
    km, scaler, cols, metrics = load_trained_clusterer()
    df = load_raw_data().head(5)
    X_c = scaler.transform(df[cols])
    labels = km.predict(X_c)
    assert len(labels) == 5
    assert (labels >= 0).all()
    assert (labels < metrics["selected_k"]).all()
