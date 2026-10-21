"""
AI-Based Business Location Selector - Streamlit Interactive Web Application
"""
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium

from src.config import (
    DEFAULT_WEIGHTS,
    DEFAULT_BUDGETS,
    BUSINESS_TYPES,
    CLUSTER_ARCHETYPES,
    POTENTIAL_COLORS,
    MODELS_DIR,
)
from src.data_loader import load_raw_data, load_geodata, get_dataset_summary
from src.model_trainer import load_trained_models
from src.clustering import load_trained_clusterer
from src.scoring_engine import rank_candidate_locations
from src.geo_utils import build_candidate_map

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Business Location Selector",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header Banner */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    padding: 24px 32px;
    border-radius: 16px;
    color: white;
    margin-bottom: 24px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
}

.main-header h1 {
    font-size: 28px;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.main-header p {
    color: #94a3b8;
    margin: 6px 0 0 0;
    font-size: 14px;
}

/* Metric Cards */
.kpi-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 18px 20px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
}

.kpi-title {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
}

.kpi-value {
    font-size: 24px;
    font-weight: 800;
    color: #0f172a;
    margin-top: 4px;
}

.kpi-sub {
    font-size: 12px;
    color: #10b981;
    font-weight: 600;
    margin-top: 2px;
}

/* Cluster Archetype Card */
.cluster-card {
    border-radius: 12px;
    padding: 16px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    margin-bottom: 12px;
}

/* Disclaimer Footer */
.disclaimer-box {
    background: #fefce8;
    border: 1px solid #fef08a;
    border-radius: 12px;
    padding: 14px 18px;
    color: #854d0e;
    font-size: 13px;
    margin-top: 32px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------
@st.cache_data
def get_data():
    return load_raw_data()

@st.cache_resource
def get_models():
    return load_trained_models()

@st.cache_resource
def get_clusterer():
    return load_trained_clusterer()

df_raw = get_data()
models_dict, feature_pipeline, model_metrics = get_models()
kmeans_model, cluster_scaler, cluster_cols, clustering_metrics = get_clusterer()

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/map-pin.png", width=54)
    st.markdown("### **Decision Parameters**")
    
    # 1. Business Category
    business_type = st.selectbox(
        "🏢 Business Type",
        options=BUSINESS_TYPES,
        index=0,
        help="Select retail or service category under evaluation."
    )
    
    # Business Profile Insights
    cat_hints = {
        "Cafe": "☕ Requires high footfall, young demographics, and outdoor ambiance.",
        "Pharmacy": "💊 Prioritizes road accessibility, hospitals nearby, and transit.",
        "Grocery Store": "🛒 Relies on catchment residential density and parking convenience.",
        "Clothing Shop": "👗 Thrives in commercial market agglomerations with shopping traffic."
    }
    st.caption(cat_hints.get(business_type, ""))

    st.markdown("---")

    # 2. Monthly Rent Budget Slider
    default_b = DEFAULT_BUDGETS.get(business_type, 35000)
    user_budget = st.slider(
        "💰 Monthly Rent Budget (₹)",
        min_value=15000,
        max_value=85000,
        value=default_b,
        step=2000,
        help="Locations with rent exceeding this budget will receive an affordability penalty."
    )

    st.markdown("---")

    # 3. Model Selector for Demand Forecasting
    model_choice = st.selectbox(
        "🤖 Demand Prediction Model",
        options=["Random Forest", "Decision Tree", "Linear Regression"],
        index=0,
        help="Choose supervised machine learning model for demand estimation."
    )

    st.markdown("---")

    # 4. Multi-Criteria Priority Weights
    with st.expander("⚖️ Custom Priority Weights", expanded=False):
        st.caption("Adjust relative importance of each factor (normalized to 100%).")
        def_w = DEFAULT_WEIGHTS.get(business_type, DEFAULT_WEIGHTS["Cafe"])
        
        w_demand = st.slider("Customer Demand (D)", 0.0, 0.50, float(def_w["demand"]), 0.05)
        w_access = st.slider("Accessibility (A)", 0.0, 0.50, float(def_w["accessibility"]), 0.05)
        w_footfall = st.slider("Pedestrian Footfall (F)", 0.0, 0.50, float(def_w["footfall"]), 0.05)
        w_mobility = st.slider("Parking & Transit (P)", 0.0, 0.50, float(def_w["mobility"]), 0.05)
        w_comp = st.slider("Competition Opportunity (C)", 0.0, 0.50, float(def_w["competition"]), 0.05)
        w_rent = st.slider("Rent Affordability (R)", 0.0, 0.50, float(def_w["rent"]), 0.05)

        total_weight = w_demand + w_access + w_footfall + w_mobility + w_comp + w_rent
        st.markdown(f"**Total Weight:** `{total_weight:.2f}` (Auto-normalized)")

        custom_weights = {
            "demand": w_demand,
            "accessibility": w_access,
            "footfall": w_footfall,
            "mobility": w_mobility,
            "competition": w_comp,
            "rent": w_rent
        }

    # Locality Filter
    all_localities = sorted(df_raw["location_name"].unique())
    selected_localities = st.multiselect(
        "📍 Filter Localities",
        options=all_localities,
        default=[],
        help="Leave empty to evaluate all 18 Bhopal candidate zones."
    )

# ---------------------------------------------------------
# Dynamic Prediction & Ranking Computation
# ---------------------------------------------------------
# Filter raw data for chosen category
df_cat = df_raw[df_raw["business_type"] == business_type].copy()
if selected_localities:
    df_cat = df_cat[df_cat["location_name"].isin(selected_localities)].copy()

if df_cat.empty:
    st.warning("No candidate records found for the selected filter combination.")
    st.stop()

# Supervised Demand Prediction
selected_model = models_dict.get(model_choice, models_dict["Random Forest"])
X_features = feature_pipeline.transform(df_cat)
predicted_demands = selected_model.predict(X_features)

# Unsupervised Cluster Assignment
X_cluster_scaled = cluster_scaler.transform(df_cat[cluster_cols])
cluster_ids = kmeans_model.predict(X_cluster_scaled)
df_cat["cluster_id"] = cluster_ids
df_cat["cluster_name"] = [
    CLUSTER_ARCHETYPES.get(cid, {}).get("name", f"Cluster {cid}")
    for cid in cluster_ids
]

# Run Scoring Engine
df_ranked = rank_candidate_locations(
    df=df_cat,
    business_type=business_type,
    budget=user_budget,
    custom_weights=custom_weights,
    predicted_demands=predicted_demands
)

# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="main-header">
        <h1>AI-Based Business Location Selector</h1>
        <p>Intelligent Multi-Criteria Decision Support & Geospatial Analytics for Commercial Real Estate in Bhopal</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Top KPI Indicator Cards
# ---------------------------------------------------------
top_cand = df_ranked.iloc[0]
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">🏆 Top Recommendation</div>
            <div class="kpi-value">{top_cand['location_name']}</div>
            <div class="kpi-sub">Suitability Score: {top_cand['suitability_score']:.1f} / 100</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi2:
    rent_diff = user_budget - top_cand['monthly_rent']
    diff_text = f"₹{abs(rent_diff):,} {'under budget' if rent_diff >= 0 else 'OVER budget'}"
    diff_color = "#10b981" if rent_diff >= 0 else "#ef4444"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">💵 Top Location Rent</div>
            <div class="kpi-value">₹{int(top_cand['monthly_rent']):,}</div>
            <div class="kpi-sub" style="color: {diff_color};">{diff_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">📈 Projected Daily Demand</div>
            <div class="kpi-value">{top_cand['predicted_demand']:.0f} <span style="font-size: 14px; font-weight: 500;">cust/day</span></div>
            <div class="kpi-sub">Model: {model_choice}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">🏷️ Market Archetype</div>
            <div class="kpi-value" style="font-size: 18px; line-height: 1.3;">{top_cand['cluster_name']}</div>
            <div class="kpi-sub">Footfall: {int(top_cand['daily_footfall']):,} / day</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br/>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Tabs Layout
# ---------------------------------------------------------
tab_map, tab_table, tab_charts, tab_clusters, tab_models, tab_simulator = st.tabs([
    "🗺️ Interactive Location Map",
    "📋 Candidate Rankings",
    "📊 Comparative Visualizations",
    "🧩 Market Clusters (K-Means)",
    "⚙️ Model Performance & Benchmarks",
    "🧪 What-If Location Simulator"
])

# ---------------------------------------------------------
# Tab 1: Interactive Location Map
# ---------------------------------------------------------
with tab_map:
    st.markdown("#### **Geospatial Location Suitability Across Bhopal**")
    st.caption("Markers are color-coded: 🟢 Green = Top Tier (Score ≥ 75), 🔵 Blue = Moderate (55–74), 🔴 Red = Caution (< 55). Click any marker for detailed commercial attributes.")

    map_col, info_col = st.columns([3, 1])

    with map_col:
        folium_map = build_candidate_map(df_ranked, top_n=20, highlight_rank_1=True)
        st_folium(
            folium_map,
            use_container_width=True,
            height=520,
            returned_objects=[]
        )

    with info_col:
        st.markdown(f"### **#1 Pick: {top_cand['location_name']}**")
        st.markdown(f"**Score:** `{top_cand['suitability_score']:.1f}` | **Rank:** `#1`")
        st.markdown(f"**Archetype:** *{top_cand['cluster_name']}*")
        st.markdown(f"- **Monthly Rent:** ₹{int(top_cand['monthly_rent']):,}")
        st.markdown(f"- **Est. Footfall:** {int(top_cand['daily_footfall']):,} pedestrians")
        st.markdown(f"- **Competitors Nearby:** {int(top_cand['competitor_count'])}")
        st.markdown(f"- **Road Access:** {int(top_cand['accessibility_score'])}/10")
        st.markdown(f"- **Transit & Parking:** {int(top_cand['public_transport_score'])}/10, {int(top_cand['parking_score'])}/10")
        st.info(f"💡 {top_cand['trade_off_explanation']}")

# ---------------------------------------------------------
# Tab 2: Candidate Rankings Table & Export
# ---------------------------------------------------------
with tab_table:
    st.markdown("#### **Ranked Candidate Locations**")
    st.caption(f"Showing evaluation for **{len(df_ranked)}** candidate sites matching **{business_type}** with budget **₹{user_budget:,}**.")

    display_cols = [
        "rank",
        "location_name",
        "suitability_score",
        "predicted_demand",
        "monthly_rent",
        "daily_footfall",
        "competitor_count",
        "sub_rent_affordability",
        "sub_access",
        "cluster_name",
        "trade_off_explanation"
    ]

    col_rename = {
        "rank": "Rank",
        "location_name": "Location",
        "suitability_score": "Score (0-100)",
        "predicted_demand": "Proj. Demand",
        "monthly_rent": "Monthly Rent (₹)",
        "daily_footfall": "Footfall",
        "competitor_count": "Competitors",
        "sub_rent_affordability": "Affordability (R)",
        "sub_access": "Access (A)",
        "cluster_name": "Cluster Archetype",
        "trade_off_explanation": "Strengths & Trade-offs"
    }

    df_display = df_ranked[display_cols].rename(columns=col_rename)

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score (0-100)": st.column_config.ProgressColumn(
                "Suitability Score",
                help="Final weighted suitability score [0, 100]",
                format="%.1f",
                min_value=0,
                max_value=100,
            ),
            "Monthly Rent (₹)": st.column_config.NumberColumn(
                "Monthly Rent",
                format="₹%d"
            )
        }
    )

    # CSV Download Button
    csv_bytes = df_ranked.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Full Ranking Report as CSV",
        data=csv_bytes,
        file_name=f"business_location_ranking_{business_type.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------
# Tab 3: Comparative Visualizations (Plotly)
# ---------------------------------------------------------
with tab_charts:
    st.markdown("#### **Multi-Attribute Trade-off & Comparison Visualizations**")

    c1, c2 = st.columns(2)

    with c1:
        # 1. Radar Chart: Top 3 Candidates
        st.markdown("##### 🕸️ Multi-Attribute Radar Comparison (Top 3)")
        top3 = df_ranked.head(3)
        categories = ["Demand (D)", "Access (A)", "Footfall (F)", "Mobility (P)", "Competition (C)", "Rent Affordability (R)"]

        fig_radar = go.Figure()
        colors = ["#10b981", "#3b82f6", "#f59e0b"]

        for idx, (_, r) in enumerate(top3.iterrows()):
            vals = [
                r["sub_demand"],
                r["sub_access"],
                r["sub_footfall"],
                r["sub_mobility"],
                r["sub_competition"],
                r["sub_rent_affordability"],
            ]
            vals.append(vals[0])  # close polygon
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill="toself",
                name=f"#{r['rank']} {r['location_name']} ({r['suitability_score']:.1f})",
                line_color=colors[idx % len(colors)]
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            margin=dict(l=40, r=40, t=30, b=30),
            height=380
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with c2:
        # 2. Scatter Plot: Monthly Rent vs Projected Demand
        st.markdown("##### ⚖️ Monthly Rent vs. Projected Demand by Cluster")
        fig_scatter = px.scatter(
            df_ranked,
            x="monthly_rent",
            y="predicted_demand",
            color="cluster_name",
            size="daily_footfall",
            hover_name="location_name",
            hover_data={"monthly_rent": ":₹", "predicted_demand": ":.0f", "suitability_score": ":.1f", "daily_footfall": True},
            labels={"monthly_rent": "Monthly Rent (₹)", "predicted_demand": "Projected Demand (Cust/Day)", "cluster_name": "Cluster Archetype"},
            color_discrete_sequence=["#10b981", "#3b82f6", "#8b5cf6", "#f59e0b"]
        )
        # Add budget line
        fig_scatter.add_vline(x=user_budget, line_dash="dash", line_color="#ef4444", annotation_text=f"Budget: ₹{user_budget:,}")
        fig_scatter.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # 3. Score Breakdown Bar Chart for Top 5
    st.markdown("##### 📊 Score Component Breakdown for Top 5 Candidates")
    top5 = df_ranked.head(5)
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="Demand (D)", x=top5["location_name"], y=top5["sub_demand"] * custom_weights["demand"], marker_color="#38bdf8"))
    fig_bar.add_trace(go.Bar(name="Access (A)", x=top5["location_name"], y=top5["sub_access"] * custom_weights["accessibility"], marker_color="#818cf8"))
    fig_bar.add_trace(go.Bar(name="Footfall (F)", x=top5["location_name"], y=top5["sub_footfall"] * custom_weights["footfall"], marker_color="#34d399"))
    fig_bar.add_trace(go.Bar(name="Mobility (P)", x=top5["location_name"], y=top5["sub_mobility"] * custom_weights["mobility"], marker_color="#fbbf24"))
    fig_bar.add_trace(go.Bar(name="Competition (C)", x=top5["location_name"], y=top5["sub_competition"] * custom_weights["competition"], marker_color="#f472b6"))
    fig_bar.add_trace(go.Bar(name="Rent Affordability (R)", x=top5["location_name"], y=top5["sub_rent_affordability"] * custom_weights["rent"], marker_color="#a78bfa"))

    fig_bar.update_layout(
        barmode="stack",
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Candidate Location",
        yaxis_title="Weighted Points Contribution"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------
# Tab 4: Market Clusters (K-Means)
# ---------------------------------------------------------
with tab_clusters:
    st.markdown("#### **Unsupervised Market Segmentation (K-Means Clustering)**")
    st.caption("K-Means groups candidate locations into 4 distinct commercial archetypes based on Rent, Footfall, Competition, Accessibility, and Catchment Population.")

    # Cluster Cards
    c_cols = st.columns(4)
    profiles = clustering_metrics.get("cluster_profiles", {})

    for idx, (cid, pdata) in enumerate(profiles.items()):
        badge_color = CLUSTER_ARCHETYPES.get(int(cid), {}).get("badge_color", "#3b82f6")
        with c_cols[idx % 4]:
            st.markdown(
                f"""
                <div class="cluster-card" style="border-top: 4px solid {badge_color};">
                    <strong style="color: {badge_color}; font-size: 14px;">{pdata['name']}</strong>
                    <div style="font-size: 12px; color: #64748b; margin-top: 4px;">{pdata['count']} Locations ({pdata['pct']}%)</div>
                    <hr style="margin: 8px 0; border: none; border-top: 1px solid #e2e8f0;"/>
                    <div style="font-size: 12px; line-height: 1.6;">
                        • <strong>Avg Rent:</strong> ₹{pdata['mean_rent']:,.0f}<br/>
                        • <strong>Avg Footfall:</strong> {pdata['mean_footfall']:,.0f}/day<br/>
                        • <strong>Avg Competitors:</strong> {pdata['mean_competitors']:.1f}<br/>
                        • <strong>Avg Access:</strong> {pdata['mean_accessibility']:.1f}/10<br/>
                        • <strong>Avg Demand:</strong> {pdata['mean_demand']:.0f} cust/day
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    k1, k2 = st.columns(2)
    with k1:
        st.markdown("##### 📉 Elbow Method (Inertia vs. k)")
        k_vals = clustering_metrics["k_range"]
        inertias = clustering_metrics["inertias"]
        fig_elbow = px.line(x=k_vals, y=inertias, markers=True, labels={"x": "Number of Clusters (k)", "y": "Within-Cluster Sum of Squares (Inertia)"})
        fig_elbow.add_vline(x=clustering_metrics["selected_k"], line_dash="dash", line_color="#10b981", annotation_text="Selected k=4")
        fig_elbow.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_elbow, use_container_width=True)

    with k2:
        st.markdown("##### 🎯 Silhouette Analysis vs. k")
        silhs = clustering_metrics["silhouette_scores"]
        fig_sil = px.bar(x=k_vals, y=silhs, labels={"x": "Number of Clusters (k)", "y": "Silhouette Score"}, color=silhs, color_continuous_scale="Blues")
        fig_sil.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
        st.plotly_chart(fig_sil, use_container_width=True)

# ---------------------------------------------------------
# Tab 5: Model Performance & Benchmarks
# ---------------------------------------------------------
with tab_models:
    st.markdown("#### **Supervised Regression Model Benchmarks**")
    st.caption("Comparison of baseline Linear Regression, interpretable Decision Tree, and ensemble Random Forest Regressor trained to forecast customer demand.")

    # Metrics Table
    bench_data = []
    for mname, mval in model_metrics.get("metrics", {}).items():
        bench_data.append({
            "Algorithm": mname,
            "Train R²": f"{mval['train_r2']:.3f}",
            "Test R²": f"{mval['test_r2']:.3f}",
            "5-Fold CV R²": f"{mval['cv_r2_mean']:.3f} (±{mval['cv_r2_std']:.3f})",
            "Test MAE": f"{mval['test_mae']:.2f} cust/day",
            "Test RMSE": f"{mval['test_rmse']:.2f} cust/day",
        })

    st.table(pd.DataFrame(bench_data))

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("##### 🌲 Random Forest Feature Importance (MDI)")
        feat_imp = model_metrics.get("feature_importances", {})
        df_imp = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"]).head(10)
        fig_imp = px.bar(
            df_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis"
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_imp, use_container_width=True)

    with m_col2:
        st.markdown("##### 💡 Model Comparison Takeaway")
        st.markdown(
            """
            - **Linear Regression Baseline:** Yields an out-of-sample $R^2 \approx 0.72$, proving strong linear correlation between footfall, accessibility, and demand.
            - **Decision Tree:** Provides hierarchical split rules but suffers from higher variance on unseen data ($R^2 \approx 0.65$).
            - **Random Forest Regressor:** Delivers strong generalization ($R^2 \approx 0.71$, 5-Fold CV $R^2 \approx 0.79$) and minimizes large outlier errors through bootstrap aggregation.
            - **Top Predictive Features:** Daily footfall, catchment population, road accessibility, and mobility ratios are the strongest drivers of retail demand.
            """
        )

# ---------------------------------------------------------
# Tab 6: What-If Location Simulator
# ---------------------------------------------------------
with tab_simulator:
    st.markdown("#### **Interactive 'What-If' Location Simulator**")
    st.caption("Input custom attributes for an unlisted commercial storefront in Bhopal to evaluate its predicted demand, cluster archetype, and suitability score in real time.")

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        sim_name = st.text_input("Candidate Storefront Name", value="Custom Plaza 101")
        sim_rent = st.number_input("Expected Monthly Rent (₹)", min_value=10000, max_value=120000, value=32000, step=1000)
        sim_pop = st.number_input("Catchment Population", min_value=5000, max_value=80000, value=35000, step=1000)
        sim_footfall = st.number_input("Estimated Daily Footfall", min_value=100, max_value=5000, value=1600, step=50)
        sim_comp = st.slider("Competitor Count (within 1 km)", 0, 20, 5)

    with sim_col2:
        sim_access = st.slider("Road Accessibility Score (1-10)", 1, 10, 8)
        sim_parking = st.slider("Parking Score (1-10)", 1, 10, 7)
        sim_transit = st.slider("Public Transit Access Score (1-10)", 1, 10, 8)
        sim_income = st.selectbox("Local Purchasing Power", ["Low", "Medium", "High"], index=1)
        sim_fac = st.selectbox("Anchor Facility Nearby", ["Offices", "Market", "College", "Hospital", "Residential", "Mixed"], index=0)

    if st.button("🚀 Evaluate Custom Candidate Location", type="primary"):
        # Construct single-row DataFrame
        df_sim = pd.DataFrame([{
            "location_name": sim_name,
            "business_type": business_type,
            "monthly_rent": sim_rent,
            "population": sim_pop,
            "competitor_count": sim_comp,
            "daily_footfall": sim_footfall,
            "accessibility_score": sim_access,
            "parking_score": sim_parking,
            "public_transport_score": sim_transit,
            "income_level": sim_income,
            "nearby_facilities": sim_fac,
            "target_demand": 100.0  # Placeholder
        }])

        # Predict demand
        X_sim_feat = feature_pipeline.transform(df_sim)
        sim_pred_demand = selected_model.predict(X_sim_feat)[0]

        # Predict cluster
        X_sim_clus = cluster_scaler.transform(df_sim[cluster_cols])
        sim_cid = int(kmeans_model.predict(X_sim_clus)[0])
        sim_cname = CLUSTER_ARCHETYPES.get(sim_cid, {}).get("name", f"Cluster {sim_cid}")

        # Rank / score
        df_sim_ranked = rank_candidate_locations(
            df=df_sim,
            business_type=business_type,
            budget=user_budget,
            custom_weights=custom_weights,
            predicted_demands=np.array([sim_pred_demand])
        )
        sim_res = df_sim_ranked.iloc[0]

        st.success(f"Evaluation Completed for **{sim_name}**!")

        res1, res2, res3 = st.columns(3)
        res1.metric("Final Suitability Score", f"{sim_res['suitability_score']:.1f} / 100")
        res2.metric("Projected Daily Demand", f"{sim_pred_demand:.0f} customers/day")
        res3.metric("Assigned Cluster", sim_cname)

        st.info(f"💡 **Analysis:** {sim_res['trade_off_explanation']}")

# ---------------------------------------------------------
# Ethical Disclosure & Academic Disclaimer Footer
# ---------------------------------------------------------
st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ Academic Integrity & Data Ethics Notice:</strong><br/>
        This project utilizes a synthetic candidate location dataset modeled on the urban commercial geography of Bhopal, Madhya Pradesh, India.
        The machine learning models and suitability algorithms demonstrate a rigorous end-to-end data science and decision-support workflow.
        A high suitability score represents strong alignment with user constraints and empirical indicators, not a certified guarantee of commercial profit.
        In commercial deployments, candidate data must be validated with primary pedestrian traffic sensors, certified lease contracts, and audited municipal filings.
    </div>
    """,
    unsafe_allow_html=True
)
