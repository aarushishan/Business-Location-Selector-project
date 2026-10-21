# Product Requirements Document (PRD)
## AI-Based Business Location Selector Using Machine Learning

- **Project Title:** AI-Based Business Location Selector Using Machine Learning
- **Domain:** Machine Learning, Geospatial Analytics, Data Analysis, and Multi-Criteria Decision Support
- **Document Version:** 1.0.0
- **Status:** Approved / Ready for Implementation (Awaiting User "start" signal)

---

## 1. Executive Summary & Vision
Selecting the optimal physical location for a commercial venture (such as a café, pharmacy, grocery store, or apparel boutique) is one of the most critical determinants of business success or failure. Traditional decision-making frequently relies on intuition, fragmented market observations, or misleading rules of thumb (e.g., assuming high footfall automatically guarantees profitability despite prohibitive rent, or misinterpreting dense competition as either pure saturation or viable demand).

The **AI-Based Business Location Selector** is an intelligent decision-support system designed to evaluate, rank, and visualize candidate business locations across urban centers (with Bhopal as the primary reference geography). By harmonizing supervised machine learning (regression demand forecasting), unsupervised machine learning (K-Means cluster profiling), geospatial mapping, and a transparent multi-criteria suitability scoring formula, the platform empowers entrepreneurs and retail planners with data-driven, auditable location insights.

---

## 2. Problem Statement
Entrepreneurs face severe multi-dimensional trade-offs when selecting commercial real estate:
1. **Demand vs. Cost:** High-density prime locations come with exorbitant rent that may compress operating margins. Conversely, cheap rent frequently corresponds to poor visibility and inadequate customer footfall.
2. **Competition Ambiguity:** Competitor density can be a double-edged sword—it may signal an established, thriving commercial cluster (agglomeration economy) or an oversaturated market where price wars cannibalize revenue.
3. **Multi-Factor Complexity:** Physical accessibility, parking availability, public transit connectivity, catchment population demographics, and purchasing power interact in non-linear ways that human heuristics fail to evaluate accurately.
4. **Data Opacity:** Decision-makers need explainable recommendations rather than opaque black-box recommendations, understanding precisely *why* a particular location was recommended and what compromises it entails.

---

## 3. Target Audience & Personas
- **Small Business Owners & Franchisees:** Seeking affordable candidate sites that maximize footfall while staying within monthly rental ceilings.
- **Retail Real Estate Analysts:** Requiring objective clustering and demand benchmarks to advise commercial tenants.
- **Academic Evaluators & Students:** Demanding rigorous end-to-end ML workflows, disciplined model benchmarking, feature engineering, and academic defensibility during project reviews and viva examinations.

---

## 4. Key Goals & System Objectives
- **Identify Core Influencing Factors:** Quantify the impact of rent, population, footfall, competitor density, transit scores, and local income on business viability.
- **Comparative Supervised Modeling:** Train, evaluate, and benchmark multiple regression algorithms (Linear Regression, Decision Tree, Random Forest) to predict customer demand / sales.
- **Unsupervised Candidate Profiling:** Segment candidate locations into interpretable clusters (e.g., *Affordable / High-Footfall*, *Premium Core*, *High-Cost / Saturation Risk*) using K-Means clustering and Silhouette analysis.
- **Transparent Multi-Criteria Scoring:** Compute an explainable, normalized Suitability Index ($S \in [0, 100]$) with customizable stakeholder weights.
- **Interactive Geospatial Dashboard:** Provide a full-featured Streamlit application featuring candidate filters, interactive Folium maps, radar comparison charts, what-if sensitivity sliders, and CSV report export.

---

## 5. Scope of Features & Functional Requirements

### Module 1: Data Ingestion & Preprocessing
- **FR 1.1:** Ingest and parse candidate location datasets (`business_location_synthetic_dataset.csv`) containing demographic, economic, and accessibility attributes.
- **FR 1.2:** Handle missing values, outliers, duplicate records, and encode categorical variables (`income_level`, `nearby_facilities`, `business_type`).
- **FR 1.3:** Perform automated feature scaling (StandardScaler and MinMax normalizations) for ML models and scoring algorithms.

### Module 2: Exploratory Data Analysis (EDA) & Diagnostics
- **FR 2.1:** Distribution plots (histograms, KDEs) for rent, population, footfall, and competitor count.
- **FR 2.2:** Correlation heatmaps identifying collinearity between accessibility, transit, footfall, and target demand.
- **FR 2.3:** Business-type segmentation analysis (e.g., how optimal metrics differ between a Café and a Pharmacy).

### Module 3: Supervised Machine Learning (Demand Prediction)
- **FR 3.1:** Implement baseline **Linear Regression** for benchmark linear interpretability.
- **FR 3.2:** Implement **Decision Tree Regressor** to extract hierarchical decision rules.
- **FR 3.3:** Implement **Random Forest Regressor** to model complex non-linear feature interactions and resist overfitting.
- **FR 3.4:** Model validation using 80/20 train-test splits and 5-fold cross-validation.
- **FR 3.5:** Compute standard regression evaluation metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Coefficient of Determination ($R^2$)
- **FR 3.6:** Extract and visualize Feature Importance rankings to provide explainable AI (XAI) insights.

### Module 4: Unsupervised Machine Learning (K-Means Clustering)
- **FR 4.1:** Cluster candidate locations using normalized multi-dimensional feature vectors (`monthly_rent`, `daily_footfall`, `competitor_count`, `accessibility_score`, `population`).
- **FR 4.2:** Determine optimal $k$ using the Elbow Method (Inertia curve) and Silhouette Score analysis ($k \in [2, 6]$).
- **FR 4.3:** Assign semantic labels to clusters (e.g., *Cluster 0: Budget High-Potential*, *Cluster 1: Saturated Premium*, *Cluster 2: Emerging Suburban*).

### Module 5: Multi-Criteria Suitability Scoring Engine
- **FR 5.1:** Compute normalized sub-scores ($[0, 100]$) for:
  - Demand Potential ($D$)
  - Accessibility & Walkability ($A$)
  - Footfall Traffic ($F$)
  - Parking & Public Transit ($P$)
  - Competition Opportunity ($C$)
  - Rent Affordability relative to user budget ($R$)
- **FR 5.2:** Implement the customizable weighted scoring formula:
  $$S = w_D \cdot D + w_A \cdot A + w_F \cdot F + w_P \cdot P + w_C \cdot C + w_R \cdot R$$
  where $\sum w_i = 1.00$.
- **FR 5.3:** Allow dynamic user adjustments of weights according to strategic priorities (e.g., higher rent penalty for boot-strapped startups).

### Module 6: Interactive Web Dashboard & Geospatial Visualization
- **FR 6.1:** **Sidebar Controls:** Business type selector, maximum monthly rent budget slider, priority weights customizer.
- **FR 6.2:** **Top KPI Bar:** Best candidate location, projected demand, average cluster rent, top suitability score.
- **FR 6.3:** **Ranked Candidates Table:** Interactive data table displaying ranks, individual sub-scores, final score, and cluster tag.
- **FR 6.4:** **Interactive Geospatial Map:** Folium map with color-coded markers (Green = High potential, Blue = Moderate, Orange = Low), custom popups displaying location metrics, and catchment radius circles.
- **FR 6.5:** **Comparative Visualizations:** Radar charts for multi-attribute comparison between top 3 locations; scatter plot of Rent vs. Projected Demand colored by cluster.
- **FR 6.6:** **Model Diagnostic Center:** Visual display of model metrics ($R^2$, MAE, RMSE), residual distributions, and feature importance bar chart.
- **FR 6.7:** **Export Facility:** One-click CSV/PDF-ready summary export of evaluated candidate rankings and audit parameters.

---

## 6. Non-Functional Requirements (NFRs)
- **NFR 1: Performance:** Dashboard response time for recalculating rankings upon slider adjustment must be under 500ms; full model inference under 1.5 seconds.
- **NFR 2: Modularity & Maintainability:** Clean separation of concerns between data processing (`src/data_processing.py`), ML models (`src/models.py`), scoring logic (`src/scoring.py`), and UI presentation (`app/`).
- **NFR 3: Transparency & Academic Integrity:** Explicit synthetic data disclosures placed prominently in the UI and documentation. Zero deceptive claims of guaranteed commercial profitability.
- **NFR 4: Portability:** Cross-platform compatibility on Windows, macOS, and Linux running standard Python environments without proprietary dependencies.

---

## 7. Project Deliverables Checklist
- [x] Product Requirements Document (`PRD.md`)
- [x] Technology Stack Specification (`TECH_STACK.md`)
- [x] Project Goals & Milestones (`GOALS.md`)
- [x] Engineering Rules & Guidelines (`RULES.md`)
- [x] Mathematical Methodology & Scoring Engine (`METHODOLOGY_AND_SCORING.md`)
- [x] Dataset Data Dictionary & Schema (`DATASET_DICTIONARY.md`)
- [x] Student Viva & Oral Exam Guide (`VIVA_QUESTIONS_GUIDE.md`)
- [ ] End-to-end Python codebase (to be initiated upon user confirmation: "start")
- [ ] Trained machine learning models & serialized artifacts
- [ ] Interactive Streamlit Dashboard application
- [ ] Comprehensive Project Report & Presentation deck
