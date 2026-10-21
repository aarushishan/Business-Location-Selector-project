"""
Geospatial Utilities module for interactive Folium mapping, coordinate lookup,
and spatial visualization of candidate business locations across Bhopal.
"""
from typing import Dict, Any, Optional
import folium
from folium import plugins
import pandas as pd

from src.config import BHOPAL_CENTER, DEFAULT_MAP_ZOOM, POTENTIAL_COLORS
from src.data_loader import load_geodata

def get_locality_coordinates(location_name: str, geo_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Lookup geographic coordinates for a given locality name.
    Handles partial and case-insensitive matching.
    """
    clean_name = location_name.strip()
    if clean_name in geo_dict:
        return geo_dict[clean_name]

    # Partial / case-insensitive search
    for k, v in geo_dict.items():
        if k.lower() in clean_name.lower() or clean_name.lower() in k.lower():
            return v
    return None

def build_candidate_map(
    df_ranked: pd.DataFrame,
    top_n: int = 15,
    highlight_rank_1: bool = True
) -> folium.Map:
    """
    Create an interactive Folium map populated with candidate business location markers,
    rich HTML tooltips, catchment circles, and color-coded potential badges.

    Args:
        df_ranked: Ranked DataFrame from scoring_engine.
        top_n: Number of candidate locations to display on map.
        highlight_rank_1: If True, draws an emphasized catchment circle around #1.

    Returns:
        Configured folium.Map instance.
    """
    geo_dict = load_geodata()
    fmap = folium.Map(
        location=BHOPAL_CENTER,
        zoom_start=DEFAULT_MAP_ZOOM,
        tiles=None,
        control_scale=True
    )

    # OpenStreetMap standard tile layer with proper attribution
    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        name="OpenStreetMap",
        max_zoom=19,
        control=False
    ).add_to(fmap)

    # Ensure Leaflet resizes container dynamically when loaded
    fmap.get_root().html.add_child(folium.Element("""
    <script>
    window.addEventListener("load", function() {
        setTimeout(function() {
            window.dispatchEvent(new Event('resize'));
        }, 250);
    });
    </script>
    """))

    # Add a fullscreen button plugin
    plugins.Fullscreen(position="topright").add_to(fmap)

    # Process candidates
    df_plot = df_ranked.head(top_n)

    for _, row in df_plot.iterrows():
        loc_name = str(row["location_name"])
        geo_info = get_locality_coordinates(loc_name, geo_dict)
        if not geo_info:
            continue

        lat = geo_info["lat"]
        lon = geo_info["lon"]
        rank = int(row.get("rank", 99))
        score = float(row.get("suitability_score", 0))
        rent = int(row.get("monthly_rent", 0))
        footfall = int(row.get("daily_footfall", 0))
        demand = float(row.get("predicted_demand", row.get("target_demand", 0)))
        competitors = int(row.get("competitor_count", 0))
        cluster_name = str(row.get("cluster_name", "Commercial Candidate"))
        business_type = str(row.get("business_type", "Business"))

        # Determine color
        if score >= 75.0:
            color = "#10b981"  # Emerald Green
            tier_badge = "TOP TIER"
            badge_bg = "#ecfdf5"
            badge_fg = "#065f46"
        elif score >= 55.0:
            color = "#3b82f6"  # Blue
            tier_badge = "MODERATE"
            badge_bg = "#eff6ff"
            badge_fg = "#1e40af"
        else:
            color = "#ef4444"  # Red / Amber
            tier_badge = "CAUTION"
            badge_bg = "#fef2f2"
            badge_fg = "#991b1b"

        # HTML Popup Card
        popup_html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; min-width: 230px; font-size: 13px; line-height: 1.4; color: #1e293b;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin-bottom: 8px;">
                <span style="font-weight: 700; font-size: 15px; color: #0f172a;">#{rank} {loc_name}</span>
                <span style="background: {badge_bg}; color: {badge_fg}; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 9999px; text-transform: uppercase;">
                    {tier_badge}
                </span>
            </div>
            <div style="margin-bottom: 6px;">
                <span style="color: #64748b; font-size: 11px;">Suitability Score</span><br/>
                <strong style="font-size: 20px; color: {color};">{score:.1f}</strong> <span style="color: #94a3b8; font-size: 12px;">/ 100</span>
            </div>
            <table style="width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 6px;">
                <tr><td style="color: #64748b; padding: 2px 0;">Category:</td><td style="font-weight: 600; text-align: right;">{business_type}</td></tr>
                <tr><td style="color: #64748b; padding: 2px 0;">Monthly Rent:</td><td style="font-weight: 600; text-align: right;">₹{rent:,}</td></tr>
                <tr><td style="color: #64748b; padding: 2px 0;">Proj. Demand:</td><td style="font-weight: 600; text-align: right;">{demand:.0f} cust/day</td></tr>
                <tr><td style="color: #64748b; padding: 2px 0;">Daily Footfall:</td><td style="font-weight: 600; text-align: right;">{footfall:,}</td></tr>
                <tr><td style="color: #64748b; padding: 2px 0;">Competitors:</td><td style="font-weight: 600; text-align: right;">{competitors}</td></tr>
                <tr><td style="color: #64748b; padding: 2px 0;">Cluster:</td><td style="font-weight: 500; font-size: 11px; text-align: right; color: #475569;">{cluster_name}</td></tr>
            </table>
        </div>
        """

        # Add Marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=12 if rank == 1 else (9 if rank <= 3 else 7),
            color="#ffffff" if rank == 1 else color,
            weight=3 if rank == 1 else 2,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"#{rank} {loc_name} — Score: {score:.1f}"
        ).add_to(fmap)

        # Highlight Rank 1 with catchment radius circle (800m)
        if rank == 1 and highlight_rank_1:
            folium.Circle(
                location=[lat, lon],
                radius=900,
                color="#10b981",
                weight=1.5,
                fill=True,
                fill_color="#10b981",
                fill_opacity=0.15,
                popup="800m Catchment Radius for Top Choice"
            ).add_to(fmap)

    return fmap
