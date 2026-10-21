"""
Master Execution Pipeline for AI-Based Business Location Selector.
Executes data ingestion, model benchmarking, K-Means clustering,
and provides a quickstart entry point for the Streamlit dashboard.
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 stdout encoding for Windows console
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.data_loader import load_raw_data, get_dataset_summary
from src.model_trainer import train_and_evaluate_models
from src.clustering import evaluate_and_fit_clusters
from src.scoring_engine import rank_candidate_locations

def run_pipeline():
    print("=" * 65)
    print(" [*] AI-BASED BUSINESS LOCATION SELECTOR -- PIPELINE RUNNER")
    print("=" * 65)

    # 1. Data Ingestion & Summary
    print("\n[Step 1/4] Ingesting and validating candidate location dataset...")
    df = load_raw_data()
    summary = get_dataset_summary(df)
    print(f" [+] Successfully loaded {summary['total_records']} records across {summary['unique_locations']} Bhopal localities.")
    print(f" [+] Average Monthly Rent: Rs. {summary['rent_mean']:,.0f} | Average Daily Footfall: {summary['footfall_mean']:,.0f}")

    # 2. Model Training & Benchmarking
    print("\n[Step 2/4] Training and benchmarking Supervised Regression Models...")
    models, pipeline, model_summary = train_and_evaluate_models(df)
    for model_name, m in model_summary["metrics"].items():
        print(f"   * {model_name:<18}: Test R2 = {m['test_r2']:.3f} | Test MAE = {m['test_mae']:.1f} cust/day | Test RMSE = {m['test_rmse']:.1f}")

    # 3. K-Means Clustering
    print("\n[Step 3/4] Running K-Means clustering and market archetype profiling...")
    km, scaler, cols, cluster_summary, df_clustered = evaluate_and_fit_clusters(df, n_clusters=4)
    print(f" [+] Optimal k = {cluster_summary['selected_k']} | Mean Silhouette Score = {cluster_summary['selected_silhouette']:.3f}")
    for cid, p in cluster_summary["cluster_profiles"].items():
        print(f"   * Cluster {cid}: {p['name']:<32} ({p['count']} locations, Avg Rent: Rs. {p['mean_rent']:,.0f})")

    # 4. Suitability Scoring & Ranking Demo
    print("\n[Step 4/4] Demonstrating Suitability Ranking for Cafe with Rs. 35,000 budget...")
    ranked_cafe = rank_candidate_locations(df, business_type="Cafe", budget=35000)
    top_picks = ranked_cafe.head(3)
    print("\n[TOP 3 CANDIDATE LOCATIONS]:")
    for _, row in top_picks.iterrows():
        print(f"   #{row['rank']} {row['location_name']:<18} | Suitability Score: {row['suitability_score']:.1f}/100 | Rent: Rs. {row['monthly_rent']:,} | Proj Demand: {row['predicted_demand']:.0f}/day")

    print("\n" + "=" * 65)
    print(" [DONE] Pipeline complete! All artifacts saved in models/ directory.")
    print(" To launch the interactive dashboard, run:")
    print("     streamlit run app/app.py")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Business Location Selector Pipeline")
    parser.add_argument("--train-only", action="store_true", help="Run model training and exit")
    args = parser.parse_args()
    run_pipeline()
