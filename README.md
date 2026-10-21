# AI-Based Business Location Selector Using Machine Learning
> **An Intelligent Geospatial Decision-Support System for Retail Site Selection**  
> *Primary Geography: Bhopal, Madhya Pradesh, India*

---

## 📌 Project Overview
The **AI-Based Business Location Selector** is an academic-grade machine learning and decision-support system designed to compare and evaluate candidate locations for new commercial enterprises (such as Cafés, Pharmacies, Grocery Stores, and Clothing Boutiques). 

By synthesizing demographic catchment estimates, commercial rent, competitor density, pedestrian footfall, vehicular accessibility, transit connectivity, and local purchasing power across **Bhopal** commercial zones, the platform produces:
1. **Supervised Demand Predictions:** Benchmarking Linear Regression, Decision Tree, and Random Forest algorithms.
2. **Unsupervised Market Segmentation:** Profiling candidate zones using K-Means clustering and Silhouette analysis into 4 market archetypes.
3. **Multi-Criteria Suitability Ranking:** Computing an auditable, normalized score ($S \in [0, 100]$) with customizable stakeholder weights.
4. **Interactive Geospatial Dashboard:** A responsive Streamlit web application with interactive Folium maps, Plotly radar trade-off charts, sensitivity sliders, and CSV report export.

---

## 🏗️ Repository Architecture

```text
BUSINESS/
├── app/
│   └── app.py                                    # Interactive Streamlit Web Application
├── data/
│   ├── business_location_synthetic_dataset.csv   # Primary dataset (1,001 candidate records)
│   └── bhopal_coordinates.json                   # Geospatial centroid lookup for 18 Bhopal hubs
├── models/
│   ├── rf_regressor.joblib                       # Serialized Random Forest model artifact
│   ├── dt_regressor.joblib                       # Serialized Decision Tree model artifact
│   ├── linear_regressor.joblib                   # Serialized Linear Regression baseline artifact
│   ├── kmeans_clusterer.joblib                   # Serialized K-Means clustering model artifact
│   ├── kmeans_scaler.joblib                      # Serialized K-Means feature scaler artifact
│   ├── feature_pipeline.joblib                   # Fitted ColumnTransformer & preprocessing pipeline
│   ├── model_metrics.json                        # Regression evaluation benchmarks (MAE, RMSE, R²)
│   └── clustering_metrics.json                   # Elbow & Silhouette cluster metrics & profiles
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb        # EDA distributions, correlations, outliers
│   ├── 02_model_training_and_comparison.ipynb    # Linear vs Tree vs RF benchmarks & metrics
│   └── 03_clustering_and_scoring.ipynb           # K-Means elbow analysis & cluster labeling
├── src/
│   ├── __init__.py                               # Package initializer
│   ├── config.py                                 # Project constants, default weights, color themes
│   ├── data_loader.py                            # Safe ingestion, schema validation, data cleaning
│   ├── feature_engineering.py                    # Domain ratios, normalizers, encoding pipelines
│   ├── model_trainer.py                          # Training pipelines, hyperparameter tuning, metrics
│   ├── clustering.py                             # K-Means clustering, silhouette scoring, cluster profiling
│   ├── scoring_engine.py                         # Multi-criteria suitability formula calculation
│   └── geo_utils.py                              # Geocoding helpers and Folium map layer generators
├── tests/
│   ├── test_data_loader.py                       # Unit tests for data loading & integrity checks
│   ├── test_scoring_engine.py                    # Unit tests for weighted suitability formulas
│   └── test_models.py                            # Sanity check for model predictions & shapes
├── main.py                                       # Master CLI pipeline runner
├── requirements.txt                              # Pinned Python package dependencies
├── PRD.md                                        # Product Requirements Document
├── TECH_STACK.md                                 # Technology Stack Specification
├── GOALS.md                                      # Project Goals, Milestones & Rubrics
├── RULES.md                                      # Engineering Rules & Integrity Guidelines
├── METHODOLOGY_AND_SCORING.md                    # Math formulation, algorithms & pipeline docs
├── DATASET_DICTIONARY.md                         # Detailed schema & variable metadata
└── VIVA_QUESTIONS_GUIDE.md                       # Comprehensive oral exam & viva preparation guide
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup
Clone or navigate to the workspace directory:
```bash
python -m pip install -r requirements.txt
```

### 2. Run Master Machine Learning Pipeline
Trains all regression models, runs K-Means clustering, and displays demo rankings:
```bash
python main.py
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run app/app.py
```
Then open your browser at `http://localhost:8501`.

### 4. Run Automated Test Suite
```bash
python -m pytest tests/ -v
```

---

## 📊 Supervised Machine Learning Benchmark Summary

| Model | Train $R^2$ | Test $R^2$ | 5-Fold CV $R^2$ | Test MAE | Test RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | 0.826 | **0.722** | 0.818 (±0.026) | 15.9 cust/day | 19.9 cust/day |
| **Decision Tree (CART)** | 0.867 | 0.649 | 0.684 (±0.024) | 17.7 cust/day | 22.3 cust/day |
| **Random Forest Regressor** | **0.947** | **0.707** | **0.788** (±0.016) | **16.1 cust/day** | **20.4 cust/day** |

---

## 🧩 Market Archetypes (Unsupervised K-Means)

- **Cluster 0 — Affordable High-Opportunity Hub:** Dense pedestrian flow and high customer demand efficiency.
- **Cluster 1 — Balanced Commercial Core:** High-traffic zone with solid mobility, moderate rents, and healthy demand.
- **Cluster 2 — Prime High-Cost Corridor:** Central commercial district with high visibility, requiring higher operating margins.
- **Cluster 3 — Suburban / Developing Market:** Lower rent and low competitor saturation, ideal for community convenience retail.

---

## 📐 Suitability Score Master Formula

$$S = w_D \cdot D + w_A \cdot A + w_F \cdot F + w_P \cdot P + w_C \cdot C + w_R \cdot R$$

- $D$ = Normalized Customer Demand Score $[0, 100]$
- $A$ = Road Accessibility Score $[0, 100]$
- $F$ = Pedestrian Footfall Score $[0, 100]$
- $P$ = Mobility & Parking Index $[0, 100]$
- $C$ = Competition Opportunity Score $[0, 100]$
- $R$ = Piecewise Budget Affordability Score $[0, 100]$

Weights default to category-specific benchmarks (e.g., Café: $0.30D + 0.20A + 0.15F + 0.15P + 0.10C + 0.10R$) and dynamically adjust in real time.

---

## 📖 Complete Documentation Index

- [PRD.md](file:///c:/Users/hp/OneDrive/Documents/BUSINESS/PRD.md)
- [TECH_STACK.md](file:///c:/Users/hp/OneDrive/Documents/BUSINESS/TECH_STACK.md)
- [GOALS.md](file:///c:/Users/hp/OneDrive/Documents/BUSINESS/GOALS.md)
- [RULES.md](file:///c:/Users/hp/OneDrive/Documents/BUSINESS/RULES.md)
- [METHODOLOGY_AND_SCORING.md](file:///c:/Users/hp/OneDrive/Documents/BUSINESS/METHODOLOGY_AND_SCORING.md)
- [DATASET_DICTIONARY.md](file:///c:/Users/hp/OneDrive/Documents/BUSINESS/DATASET_DICTIONARY.md)
- [VIVA_QUESTIONS_GUIDE.md](file:///c:/Users/hp/OneDrive/Documents/BUSINESS/VIVA_QUESTIONS_GUIDE.md)
