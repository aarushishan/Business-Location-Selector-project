# Project Goals, Milestones & Deliverables Roadmap
## AI-Based Business Location Selector Using Machine Learning

---

## 1. High-Level Vision & Purpose
The primary objective of this project is to develop an academic-grade, end-to-end Machine Learning and Geospatial Decision Support System that systematically compares, evaluates, and ranks candidate commercial locations for entrepreneurs.

By synthesizing multi-source urban data (rent, footfall, population, competitor saturation, transit accessibility, parking, and purchasing power), the system replaces ungrounded guesswork with empirical demand predictions, cluster-based archetype profiling, and transparent multi-attribute scoring.

---

## 2. Specific Learning & Technical Objectives

### A. Data Science & Exploratory Analytics
- Clean, profile, and validate a comprehensive 1,001-row candidate location dataset across Bhopal urban zones.
- Uncover statistical interactions between footfall, rent rates, demographic population size, and commercial competitor saturation.
- Engineer domain-specific indicators, including competitor density, rent-to-budget affordability ratios, and multi-modal mobility indices.

### B. Supervised Machine Learning (Demand Forecasting)
- Train and benchmark three distinct regression algorithms:
  - **Baseline:** Ordinary Least Squares (OLS) Linear Regression.
  - **Interpretable Tree:** Decision Tree Regressor (CART algorithm).
  - **Non-Linear Ensemble:** Random Forest Regressor with 100+ bagged trees.
- Evaluate model generalization on held-out test data using $R^2$, MAE, and RMSE.
- Derive and visualize feature importances to reveal the strongest drivers of commercial demand.

### C. Unsupervised Machine Learning (Market Segmentation)
- Implement K-Means clustering on normalized feature vectors.
- Quantitatively determine optimal cluster count $k$ using the Elbow method (Inertia) and Silhouette coefficient analysis.
- Assign actionable business interpretations to discovered clusters (e.g., *Budget High-Density*, *Premium Commercial Core*, *Suburban Growth Corridor*).

### D. Multi-Criteria Decision Analysis (MCDA)
- Formulate a transparent, normalized Suitability Index ($S \in [0, 100]$):
  $$S = 0.30D + 0.20A + 0.15F + 0.15P + 0.10C + 0.10R$$
- Enable dynamic weight sensitivity analysis where business owners customize weights to match their specific operating constraints.

### E. Interactive Geospatial Dashboard
- Construct a modern, intuitive Streamlit web application.
- Integrate interactive Folium geospatial mapping displaying Bhopal candidate zones with color-coded potential indicators.
- Provide radar charts, scatter trade-off plots, natural-language recommendation summaries, and one-click data export.

---

## 3. Four-Week Implementation Timeline

| Phase | Timeline | Key Tasks & Milestones | Expected Deliverables |
| :--- | :--- | :--- | :--- |
| **Week 1** | **Plan & Prepare** | - Finalize project scope and select business types (Café, Pharmacy, Grocery, Clothing).<br/>- Ingest and validate `business_location_synthetic_dataset.csv`.<br/>- Conduct Exploratory Data Analysis (EDA) and identify feature distributions.<br/>- Document data schema, ethics, and synthetic limitations. | - `PRD.md`, `TECH_STACK.md`, `DATASET_DICTIONARY.md`<br/>- Cleaned data matrices in `data/processed/`<br/>- EDA Notebook (`01_exploratory_data_analysis.ipynb`) |
| **Week 2** | **Build ML Models** | - Implement train/test split and cross-validation strategy.<br/>- Train Linear Regression, Decision Tree, and Random Forest models.<br/>- Hyperparameter tuning via GridSearchCV.<br/>- Implement K-Means clustering with Elbow and Silhouette plots.<br/>- Implement Suitability Score formula in modular Python code. | - Model training notebook (`02_model_training_and_comparison.ipynb`)<br/>- Clustering notebook (`03_clustering_and_scoring.ipynb`)<br/>- Serialized model artifacts in `models/*.joblib`<br/>- Benchmark performance comparison table |
| **Week 3** | **Develop Application** | - Construct modular Streamlit application architecture (`app/app.py`).<br/>- Implement sidebar filters (business category, budget, weight sliders).<br/>- Embed Folium geospatial map with candidate markers & popups.<br/>- Add Plotly comparison charts (radar, trade-off scatter).<br/>- Build explainability panel showing individual score components. | - Interactive Streamlit application (`app/`)<br/>- Interactive geospatial and radar visualization modules<br/>- CSV report export functionality |
| **Week 4** | **Test & Present** | - Conduct system verification, edge-case testing, and unit tests via Pytest.<br/>- Verify reproducibility across clean environments.<br/>- Prepare comprehensive academic project report and slide deck.<br/>- Rehearse viva defense questions and model trade-off justifications. | - Comprehensive Final Project Report (`reports/`)<br/>- PowerPoint Presentation (`presentation.pptx`)<br/>- Complete source code repository with `README.md`<br/>- Mastery of `VIVA_QUESTIONS_GUIDE.md` |

---

## 4. Final Deliverables Checklist

- [x] **Project Synopsis & Problem Statement:** Articulated in `PRD.md`.
- [x] **Dataset Documentation:** Comprehensive data dictionary, source disclosures, and field definitions in `DATASET_DICTIONARY.md`.
- [ ] **Data Cleaning & EDA Notebook:** `notebooks/01_exploratory_data_analysis.ipynb`.
- [ ] **Model Training & Comparison Notebook:** `notebooks/02_model_training_and_comparison.ipynb`.
- [ ] **Clustering & Segmentation Notebook:** `notebooks/03_clustering_and_scoring.ipynb`.
- [ ] **Modular Python Engine:** Complete `src/` modules for data loading, feature engineering, model training, clustering, and scoring.
- [ ] **Saved Model Artifacts:** Serialized `.joblib` files in `models/`.
- [ ] **Interactive Streamlit Web Dashboard:** Production-ready `app/app.py` with multi-page tabs, Folium maps, and Plotly charts.
- [ ] **Academic Viva & Interview Guide:** Fully structured in `VIVA_QUESTIONS_GUIDE.md`.
- [ ] **Comprehensive Project Report & Presentation Slides:** Documentation ready for university submission and oral defense.

---

## 5. Academic Evaluation Rubric

| Criterion | Weight | Distinction Criteria (Grade A+) |
| :--- | :---: | :--- |
| **Problem Definition & Methodology** | 15% | Crystal-clear understanding of the location selection dilemma; well-reasoned feature selection; transparent MCDA formulation. |
| **Data Engineering & EDA** | 20% | Rigorous handling of distributions, outlier detection, scaling, and domain feature engineering (competitor density, affordability ratios). |
| **ML Modeling & Benchmarking** | 25% | Multi-model comparison (Linear Regression vs. Decision Tree vs. Random Forest); disciplined cross-validation; accurate interpretation of MAE, RMSE, and $R^2$; K-Means with Elbow & Silhouette justification. |
| **Interactive Dashboard & UX** | 20% | Flawless, responsive Streamlit dashboard featuring Folium geospatial maps, dynamic radar charts, weight sliders, and natural-language explanations. |
| **Transparency & Viva Defense** | 20% | Honest disclosure of synthetic data limitations; ability to articulate the difference between predictive accuracy and suitability ranking; confident defense during oral examination. |
