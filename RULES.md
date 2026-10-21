# Engineering Rules, Coding Guidelines & Ethics Protocols
## AI-Based Business Location Selector Using Machine Learning

- **Scope:** All contributors, developers, and AI assistants working on this repository.
- **Enforcement Level:** Mandatory.

---

## 1. Project Execution Gate (The "Start" Rule)
> [!IMPORTANT]
> **AWAIT USER INSTRUCTION TO START IMPLEMENTATION:**
> The AI assistant and developers must complete all foundational planning, architecture, schemas, and governance markdown files first.
> **DO NOT** execute training scripts, generate boilerplate application code, or start building the final Streamlit app until the user explicitly responds with **"start"** or gives the command to begin coding.

---

## 2. Academic Integrity & Data Ethics Protocols

### Rule 2.1: Explicit Synthetic Data Disclosures
- The primary dataset (`business_location_synthetic_dataset.csv`) contains synthetic candidate records modeled on Bhopal's urban commercial zones.
- **Mandatory Disclaimer:** All documentation, UI banners, and project reports must prominently state:
  > *"Note on Data Integrity: This application utilizes a synthetic dataset designed to illustrate the end-to-end data science and machine learning workflow. Predictions demonstrate model training and comparative methodology, not certified real-world commercial outcomes. In a real-world deployment, inputs must be verified with primary footfall sensors, municipal records, and audited financial statements."*
- **Strict Prohibition:** Never claim in project reports or presentations that this model guarantees real business profitability or has certified predictive power in actual commercial markets.

### Rule 2.2: Separation of Predictive Accuracy vs. Suitability Ranking
- **Predictive ML Accuracy** ($R^2$, MAE, RMSE) evaluates how accurately a supervised model fits the training target variable (`target_demand`).
- **Suitability Ranking** ($S$) is a multi-criteria heuristic decision score incorporating user budget, transit convenience, footfall, and strategic weights.
- High suitability **does not** equal high predictive accuracy, and vice versa. Always present them as complementary, distinct analytical dimensions.

---

## 3. Code Quality & Engineering Standards

### Rule 3.1: Pythonic Standards & PEP 8
- Adhere strictly to **PEP 8** style guidelines.
- Use explicit type annotations (`typing.Dict`, `typing.List`, `typing.Optional`, `typing.Tuple`, etc.) on all function and method signatures.
- Maximum line length should be kept within 100 characters for readability.

### Rule 3.2: Modular Architecture (No Monolithic Scripts)
- Code must be decomposed into dedicated packages:
  - `src/data_loader.py`: Safe I/O, validation, missing value imputation.
  - `src/feature_engineering.py`: Normalizers, categorical encoders, interaction terms.
  - `src/model_trainer.py`: Regressors (Linear, Decision Tree, Random Forest), evaluation metrics, hyperparameter grid search.
  - `src/clustering.py`: K-Means clustering, silhouette scores, cluster naming logic.
  - `src/scoring_engine.py`: Weighted multi-criteria calculation and ranking.
  - `src/geo_utils.py`: Coordinate mapping, spatial calculations, Folium visualizers.
  - `app/`: Streamlit user interface components only. Business logic must live in `src/`, never directly in Streamlit render functions.

### Rule 3.3: Data Leakage Prevention
- Always execute `train_test_split` **prior** to fitting scalers (`StandardScaler`, `MinMaxScaler`) or encoders.
- Fit scalers strictly on the training partition ($X_{train}$) and apply `.transform()` on testing ($X_{test}$) and live inference data.
- Feature engineering must never incorporate information from future decision horizons.

### Rule 3.4: Comprehensive Docstrings & Comments
- Every module, class, and function must include a clear docstring in Google or NumPy format:
  ```python
  def calculate_suitability_score(
      demand_score: float,
      accessibility_score: float,
      footfall_score: float,
      parking_transit_score: float,
      competition_score: float,
      rent_affordability_score: float,
      weights: Optional[Dict[str, float]] = None
  ) -> float:
      """
      Calculate the weighted suitability score S for a candidate location.

      Args:
          demand_score: Normalized demand potential [0, 100].
          accessibility_score: Normalized road access score [0, 100].
          footfall_score: Normalized pedestrian footfall score [0, 100].
          parking_transit_score: Combined mobility & parking index [0, 100].
          competition_score: Normalized competitor opportunity score [0, 100].
          rent_affordability_score: Normalized rent-to-budget affordability [0, 100].
          weights: Custom weight dictionary. Defaults to standard Cafe weights.

      Returns:
          Composite suitability score S bounded between 0 and 100.
      """
  ```

---

## 4. Model Benchmarking & Evaluation Guidelines

### Rule 4.1: Mandatory Baseline Model
- An advanced machine learning model (e.g., Random Forest Regressor) must always be benchmarked against a baseline model (Linear Regression) and an interpretable single-tree model (Decision Tree).
- You must explain in the results whether the increased complexity of ensemble learning produces a statistically significant reduction in MAE/RMSE on unseen test data.

### Rule 4.2: Rigorous Cluster Validation
- K-Means must not be run with an arbitrarily chosen $k$.
- You must evaluate $k \in [2, 6]$ using:
  1. **Elbow Method:** Inertia reduction curve (within-cluster sum of squares).
  2. **Silhouette Coefficient:** Quantifying cluster separation and cohesion.
- Every cluster must have a clear business interpretation that non-technical stakeholders can understand.

---

## 5. UI/UX & Design Guidelines for the Dashboard

### Rule 5.1: Aesthetic Excellence & Polish
- Use modern typography, curated color palettes, clear hierarchy, and smooth cards.
- Avoid generic plain red/blue/green styling. Use refined status palettes (Emerald for high potential, Amber for moderate, Coral/Rose for low potential/caution).
- The dashboard must load without layout jumping or blocking computation threads. Use `@st.cache_data` and `@st.cache_resource` for heavy model loading and data processing.

### Rule 5.2: Informative Visualizations
- Interactive Folium maps must include informative marker tooltips and popups detailing location name, rent, predicted demand, and score.
- Comparison tables must offer sorting, filtering, and rank highlighting.
- Provide a dedicated explanation panel that breaks down *why* Location A ranked ahead of Location B.
