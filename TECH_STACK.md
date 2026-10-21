# Technology Stack & System Architecture Specification
## AI-Based Business Location Selector Using Machine Learning

- **Project:** AI-Based Business Location Selector
- **Target OS:** Windows 10/11, macOS, Linux (Cross-platform)
- **Primary Runtime:** Python 3.10 – 3.14
- **Interface Tier:** Streamlit Web Application with Reactive Components

---

## 1. Complete Technology Stack Matrix

| Layer / Subsystem | Technology | Version | Purpose & Rationale |
| :--- | :--- | :--- | :--- |
| **Programming Language** | **Python** | `>= 3.10` | Industry standard for data science, machine learning, and rapid prototyping. |
| **Data Manipulation** | **Pandas** | `>= 2.0.0` | High-performance tabular data structures, aggregation, filtering, and feature extraction. |
| **Numerical Computing** | **NumPy** | `>= 1.24.0` | Vectorized mathematical operations, array manipulation, and fast matrix computations. |
| **Machine Learning Core** | **Scikit-Learn** | `>= 1.3.0` | Implementations of `RandomForestRegressor`, `DecisionTreeRegressor`, `LinearRegression`, `KMeans`, `StandardScaler`, `MinMaxScaler`, and cross-validation pipelines. |
| **Interactive Dashboard** | **Streamlit** | `>= 1.30.0` | Pure-Python interactive web app framework enabling rapid deployment of reactive widgets, sliders, tables, and metric cards. |
| **Geospatial Mapping** | **Folium** & **streamlit-folium** | `>= 0.15.0` | Leaflet.js-powered interactive maps for geographic coordinate rendering, custom circle markers, and popup tooltips. |
| **Interactive Visualizations** | **Plotly** | `>= 5.18.0` | Publication-grade dynamic charts: radar charts for multi-attribute comparison, 3D/2D scatter plots for cluster visualization, and waterfall charts. |
| **Static Visualizations** | **Matplotlib** & **Seaborn** | `>= 3.8.0 / >= 0.13.0` | Generation of exploratory histograms, correlation heatmaps, box plots, and model residual plots for academic reports. |
| **Model Persistence** | **Joblib** | `>= 1.3.0` | High-efficiency serialization and deserialization of trained estimators, preprocessing pipelines, and clustering objects. |
| **Testing & Quality Assurance** | **Pytest** | `>= 7.4.0` | Automated unit tests for data cleaning pipelines, feature transformers, and suitability score edge cases. |

---

## 2. Architectural Blueprint

The application follows a clean, decoupled 4-tier modular architecture:

```mermaid
graph TD
    subgraph Data Layer
        A[Raw CSV Dataset] --> B[Data Ingestion & Cleaning]
        B --> C[Feature Engineering & Normalization]
    end

    subgraph Analytics & Modeling Layer
        C --> D1[Supervised Regressors<br/>RF, DT, Linear]
        C --> D2[Unsupervised Clustering<br/>K-Means + Silhouette]
        C --> D3[Suitability Scoring Engine<br/>Weighted Multi-Criteria]
    end

    subgraph Model Serialization
        D1 --> E[(Joblib Saved Models)]
        D2 --> E
    end

    subgraph Presentation & UI Layer (Streamlit)
        E --> F[Dashboard Controller]
        D3 --> F
        F --> G1[Candidate Ranking Table]
        F --> G2[Interactive Folium Map]
        F --> G3[Plotly Radar & Scatter Charts]
        F --> G4[Explainability & Sensitivity Panel]
        F --> G5[CSV Export Facility]
    end
```

---

## 3. Detailed Repository Layout

```text
BUSINESS/
├── data/
│   ├── business_location_synthetic_dataset.csv   # Primary dataset (1,001 candidate records)
│   ├── bhopal_coordinates_reference.json         # GeoJSON/coordinates lookup for Bhopal hubs
│   └── processed/                                # Cleaned, transformed feature matrices
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb        # Data distributions, correlations, outliers
│   ├── 02_model_training_and_comparison.ipynb    # Linear vs Tree vs RF benchmarks & metrics
│   └── 03_clustering_and_scoring.ipynb           # K-Means elbow analysis & cluster labeling
├── src/
│   ├── __init__.py                               # Package initializer
│   ├── config.py                                 # Project constants, default weights, color palettes
│   ├── data_loader.py                            # Safe ingestion, schema validation, data cleaning
│   ├── feature_engineering.py                    # Density calculations, normalizers, encoding
│   ├── model_trainer.py                          # Training pipelines, hyperparameter tuning, metrics
│   ├── clustering.py                             # K-Means clustering, silhouette scoring, cluster profiling
│   ├── scoring_engine.py                         # Multi-criteria suitability formula calculation
│   └── geo_utils.py                              # Geocoding helpers and map layer generators
├── models/
│   ├── rf_regressor.joblib                       # Serialized Random Forest model artifact
│   ├── dt_regressor.joblib                       # Serialized Decision Tree model artifact
│   ├── linear_regressor.joblib                   # Serialized Linear Regression baseline artifact
│   ├── kmeans_clusterer.joblib                   # Serialized K-Means clustering model artifact
│   └── scaler.joblib                             # Serialized StandardScaler / MinMaxScaler artifact
├── app/
│   ├── app.py                                    # Main Streamlit application entry point
│   ├── components/
│   │   ├── sidebar.py                            # User filters, business type & budget controls
│   │   ├── metrics.py                            # KPI summary indicator cards
│   │   ├── table_view.py                         # Interactive ranking & candidate comparison table
│   │   ├── map_view.py                           # Folium interactive geospatial map component
│   │   ├── charts.py                             # Plotly radar, scatter, and feature importance charts
│   │   └── explanations.py                       # Natural language explanation & trade-off rationale
│   └── styles/
│       └── modern_theme.css                      # Custom CSS styling for polished visual polish
├── tests/
│   ├── test_data_loader.py                       # Unit tests for data loading & integrity checks
│   ├── test_scoring_engine.py                    # Unit tests for weighted suitability formulas
│   └── test_models.py                            # Sanity check for model predictions & shapes
├── PRD.md                                        # Product Requirements Document
├── TECH_STACK.md                                 # Technology Stack Specification (This document)
├── GOALS.md                                      # Project Goals, Milestones & Rubrics
├── RULES.md                                      # Engineering Rules & Integrity Guidelines
├── METHODOLOGY_AND_SCORING.md                    # Math formulation, algorithms & pipeline docs
├── DATASET_DICTIONARY.md                         # Detailed schema & variable metadata
├── VIVA_QUESTIONS_GUIDE.md                       # Comprehensive oral exam & viva preparation guide
├── requirements.txt                              # Pinned Python package dependencies
└── README.md                                     # Project overview and quickstart instructions
```

---

## 4. Software Dependencies (`requirements.txt`)

```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0
streamlit>=1.30.0
folium>=0.15.0
streamlit-folium>=0.17.0
joblib>=1.3.0
pytest>=7.4.0
```

---

## 5. Development & Deployment Workflow

1. **Environment Setup:**
   ```bash
   python -m venv venv
   # Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Model Training & Artifact Generation:**
   ```bash
   python -m src.model_trainer
   python -m src.clustering
   ```

3. **Running the Streamlit Dashboard:**
   ```bash
   streamlit run app/app.py
   ```

4. **Testing Suite:**
   ```bash
   pytest tests/ -v
   ```
