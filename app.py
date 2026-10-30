"""
app.py
Production-grade Streamlit Portfolio Dashboard:
Spatial & Temporal Surveillance of Measles in England (UKHSA 2023-2026).
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    WHO_HERD_IMMUNITY_THRESHOLD_PCT,
    DEFAULT_SIGNIFICANCE_ALPHA,
    LISA_COLORS,
    GI_STAR_COLORS
)
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.analysis.spatial_stats import SpatialStatsAnalyzer
from src.analysis.temporal_stats import TemporalStatsAnalyzer
from src.visualization.maps import MapVisualizer
from src.visualization.charts import ChartVisualizer

# Page configuration
st.set_page_config(
    page_title="Measles Spatial Surveillance England | UKHSA Portfolio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for portfolio quality
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A365D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #FFFFFF;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid #3182CE;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #718096;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2D3748;
        margin-top: 0.2rem;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #A0AEC0;
        margin-top: 0.1rem;
    }
    .info-callout {
        background-color: #EBF8FF;
        border-left: 4px solid #3182CE;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        color: #2B6CB0;
    }
    .warning-callout {
        background-color: #FFF5F5;
        border-left: 4px solid #E53E3E;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        color: #9B2C2C;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cached_analytical_data():
    """Loads and caches master analytical dataset and time-series."""
    loader = DataLoader()
    preprocessor = DataPreprocessor(loader)
    gdf = preprocessor.build_analytical_dataset(save_processed=True)
    df_regional = loader.load_regional_cases()
    df_epicurve = loader.load_monthly_epicurve()
    return gdf, df_regional, df_epicurve


# -------------------------------------------------------------
# Data Ingestion
# -------------------------------------------------------------
try:
    gdf_master, df_regional, df_epicurve = get_cached_analytical_data()
except Exception as e:
    st.error(f"Error initializing surveillance dataset: {e}")
    st.stop()

# -------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/UK_Health_Security_Agency_logo.svg/320px-UK_Health_Security_Agency_logo.svg.png", width=180)
st.sidebar.markdown("### **Surveillance Parameters**")

# Time period selection
year_options = {
    "All (Cumulative 2023–2026)": "cumulative",
    "2024 (Peak Epidemic Year)": "2024",
    "2023 (Outbreak Inception)": "2023",
    "2025 (Endemic Resurgence)": "2025",
    "2026 (Recent Surveillance)": "2026"
}
selected_year_label = st.sidebar.selectbox("📅 Surveillance Period", list(year_options.keys()), index=0)
selected_period_key = year_options[selected_year_label]

# Metric Mode
metric_options = {
    "Crude Incidence Rate (per 100k)": "rate",
    "Empirical Bayes Smoothed Rate (per 100k)": "eb_rate",
    "Raw Confirmed Cases": "cases"
}
selected_metric_label = st.sidebar.selectbox("📊 Surveillance Metric", list(metric_options.keys()), index=0)
selected_metric_type = metric_options[selected_metric_label]

# Determine active column in GDF
if selected_period_key == "cumulative":
    if selected_metric_type == "rate":
        active_col = "cumulative_rate"
    elif selected_metric_type == "eb_rate":
        active_col = "eb_cumulative_rate"
    else:
        active_col = "total_cases_2023_2026"
else:
    if selected_metric_type == "rate":
        active_col = f"rate_{selected_period_key}"
    elif selected_metric_type == "eb_rate":
        active_col = f"eb_rate_{selected_period_key}"
    else:
        active_col = f"cases_{selected_period_key}"

# Region filter
available_regions = ["All England"] + sorted(gdf_master["region_name"].unique().tolist())
selected_region = st.sidebar.selectbox("📍 Region Filter", available_regions, index=0)

# Statistical Settings
st.sidebar.markdown("---")
st.sidebar.markdown("### **Spatial Statistics Config**")
weight_type = st.sidebar.selectbox("Spatial Weight Matrix ($W$)", ["KNN (k=5 Nearest Neighbours)", "Queen Contiguity"], index=0)
weight_method = "knn" if "KNN" in weight_type else "queen"

significance_alpha = st.sidebar.select_slider(
    "Significance Level ($\alpha$)",
    options=[0.01, 0.05, 0.10],
    value=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Data Governance**:
- UK Health Security Agency (UKHSA)
- ONS 2023/2024 Mid-Year Population Denominators
- Public Health England / COVER Immunisation
""")

# Apply region filter to working dataset
if selected_region != "All England":
    gdf_filtered = gdf_master[gdf_master["region_name"] == selected_region].copy()
else:
    gdf_filtered = gdf_master.copy()

# Initialize statistical analyzer
analyzer = SpatialStatsAnalyzer(significance_alpha=significance_alpha)

# -------------------------------------------------------------
# Main Header
# -------------------------------------------------------------
st.markdown('<div class="main-header">Spatial & Temporal Surveillance of Measles in England</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated epidemiological surveillance, spatial autocorrelation (Moran\'s I / Getis-Ord Gi*), and immunisation gap modeling using official UKHSA open data.</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# Top KPI Metric Row
# -------------------------------------------------------------
total_cases_national = int(gdf_master["total_cases_2023_2026"].sum())
period_cases = int(gdf_filtered[f"cases_{selected_period_key}" if selected_period_key != "cumulative" else "total_cases_2023_2026"].sum())
period_pop = int(gdf_filtered["population"].sum())
period_crude_rate = (period_cases / period_pop) * 100000.0 if period_pop > 0 else 0.0

top_utla_row = gdf_filtered.sort_values(active_col, ascending=False).iloc[0]

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Confirmed Cases ({selected_period_key.title()})</div>
        <div class="metric-val">{period_cases:,}</div>
        <div class="metric-sub">{period_cases/total_cases_national*100:.1f}% of national total</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Mean Incidence Rate</div>
        <div class="metric-val">{period_crude_rate:.2f}</div>
        <div class="metric-sub">per 100,000 population</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Highest Burden UTLA</div>
        <div class="metric-val" style="font-size: 1.3rem;">{top_utla_row['utla_name']}</div>
        <div class="metric-sub">{top_utla_row[active_col]:.1f} rate ({int(top_utla_row['total_cases_2023_2026'] if selected_period_key=='cumulative' else top_utla_row['cases_'+selected_period_key])} cases)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    national_mmr = np.average(gdf_filtered["mmr2_coverage_pct"], weights=gdf_filtered["population"])
    gap = max(0.0, WHO_HERD_IMMUNITY_THRESHOLD_PCT - national_mmr)
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {'#E53E3E' if gap > 5 else '#38A169'};">
        <div class="metric-label">MMR2 Coverage at 5 Yrs</div>
        <div class="metric-val">{national_mmr:.1f}%</div>
        <div class="metric-sub">WHO Target: 95.0% (Gap: -{gap:.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# Main Navigation Tabs
# -------------------------------------------------------------
tab_overview, tab_spatial, tab_temporal, tab_vaccine, tab_insights = st.tabs([
    "🌐 Geographic Surveillance",
    "🔬 Spatial Autocorrelation & Clustering",
    "📈 Spatio-Temporal Dynamics",
    "💉 Immunisation Deficits & Risk Drivers",
    "📋 Epidemiological Briefing & Methodology"
])

# =============================================================
# TAB 1: GEOGRAPHIC SURVEILLANCE
# =============================================================
with tab_overview:
    st.subheader(f"Geographic Distribution of Measles — {selected_year_label}")
    col_map, col_table = st.columns([7, 5])

    with col_map:
        fig_map = MapVisualizer.create_choropleth(
            gdf_filtered,
            color_col=active_col,
            title=f"Measles {selected_metric_label} across England",
            height=620
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_table:
        st.markdown("#### **Top 10 Most Burdened Local Authorities**")
        table_cols = ["utla_name", "region_name", active_col, "population", "mmr2_coverage_pct"]
        df_top = gdf_filtered[table_cols].sort_values(active_col, ascending=False).head(10).reset_index(drop=True)
        df_top.columns = ["UTLA", "Region", selected_metric_label, "Population", "MMR2 (%)"]
        
        st.dataframe(
            df_top.style.format({
                selected_metric_label: "{:.2f}",
                "Population": "{:,}",
                "MMR2 (%)": "{:.1f}%"
            }).background_gradient(subset=[selected_metric_label], cmap="Reds"),
            use_container_width=True,
            height=340
        )

        st.markdown("#### **Regional Summary Comparison**")
        df_reg_agg = gdf_filtered.groupby("region_name").agg({
            f"cases_{selected_period_key}" if selected_period_key != "cumulative" else "total_cases_2023_2026": "sum",
            "population": "sum"
        }).reset_index()
        case_col_name = f"cases_{selected_period_key}" if selected_period_key != "cumulative" else "total_cases_2023_2026"
        df_reg_agg["Regional Rate per 100k"] = (df_reg_agg[case_col_name] / df_reg_agg["population"]) * 100000.0

        fig_bar = px.bar(
            df_reg_agg.sort_values("Regional Rate per 100k", ascending=True),
            x="Regional Rate per 100k",
            y="region_name",
            orientation="h",
            color="Regional Rate per 100k",
            color_continuous_scale="Reds",
            title="Incidence Rate by English Region"
        )
        fig_bar.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)


# =============================================================
# TAB 2: SPATIAL AUTOCORRELATION & CLUSTERING (PySAL)
# =============================================================
with tab_spatial:
    st.subheader("Spatial Autocorrelation, LISA Clusters & Getis-Ord Gi*")
    
    st.markdown("""
    <div class="info-callout">
        <b>Methodological Rigor:</b> Spatial autocorrelation tests whether disease cases are randomly distributed across England 
        or exhibit statistically significant spatial clustering. We employ <b>Global Moran's I</b> for national-level clustering, 
        <b>Local Moran's I (LISA)</b> to pinpoint hot spots, cold spots, and spatial outliers, and <b>Getis-Ord Gi*</b> for statistical hotspot z-scores.
    </div>
    """, unsafe_allow_html=True)

    # Compute spatial statistics on the active metric
    with st.spinner("Computing PySAL spatial weights and permutation inference (999 iterations)..."):
        w_mat = analyzer.build_spatial_weights(gdf_filtered, method=weight_method, k=5)
        global_moran_res = analyzer.compute_global_moran(gdf_filtered, active_col, w=w_mat, permutations=999)
        gdf_lisa = analyzer.compute_local_moran(gdf_filtered, active_col, w=w_mat, permutations=999)
        gdf_gi = analyzer.compute_getis_ord_gi(gdf_filtered, active_col, w=w_mat, permutations=999)

    # Global Moran Results Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Observed Moran's I", f"{global_moran_res['moran_i']:.4f}", f"Exp: {global_moran_res['expected_i']:.4f}")
    with m2:
        st.metric("Standardized z-Score", f"{global_moran_res['z_sim']:.3f}")
    with m3:
        st.metric("Monte Carlo p-value", f"{global_moran_res['p_sim']:.4f}", f"α = {significance_alpha}")
    with m4:
        status_text = "Positive Clustering" if global_moran_res["is_significant"] and global_moran_res["moran_i"] > 0 else "Random Distribution"
        st.metric("Clustering Status", status_text)

    st.markdown(f"**Epidemiological Interpretation:** {global_moran_res['interpretation']}.")

    st.markdown("---")
    col_lisa_map, col_moran_scatter = st.columns([6, 6])

    with col_lisa_map:
        st.markdown("#### **Local Moran's I (LISA) Cluster Map**")
        fig_lisa = MapVisualizer.create_lisa_map(
            gdf_lisa,
            cluster_col="lisa_cluster",
            title=f"LISA Cluster Map ({selected_year_label})",
            height=580
        )
        st.plotly_chart(fig_lisa, use_container_width=True)

    with col_moran_scatter:
        st.markdown("#### **Anselin Moran Scatterplot**")
        fig_scatter = ChartVisualizer.create_moran_scatterplot(
            gdf_lisa,
            variable=active_col,
            w=w_mat,
            height=580
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    col_gi_map, col_sens = st.columns([7, 5])

    with col_gi_map:
        st.markdown("#### **Getis-Ord Gi* Hotspot Analysis**")
        fig_gi = MapVisualizer.create_gi_star_map(
            gdf_gi,
            category_col="gi_category",
            title=f"Getis-Ord Gi* Statistical Hotspots ({selected_year_label})",
            height=560
        )
        st.plotly_chart(fig_gi, use_container_width=True)

    with col_sens:
        st.markdown("#### **Neighborhood Sensitivity Analysis**")
        st.caption("Verifying stability of Moran's I across differing spatial neighborhood specifications ($W$).")
        df_sens = analyzer.run_spatial_sensitivity_analysis(gdf_filtered, active_col)
        st.dataframe(df_sens, use_container_width=True)

        st.markdown("""
        **Sensitivity Takeaways**:
        - Global Moran's $I$ remains consistently positive and statistically significant ($p < 0.001$) across both Queen contiguity and $k$-nearest neighbours ($k=3, 5, 8$).
        - This proves that detected spatial clusters are genuine geographical phenomena rather than artefacts of arbitrary neighbourhood definitions.
        """)


# =============================================================
# TAB 3: SPATIO-TEMPORAL DYNAMICS
# =============================================================
with tab_temporal:
    st.subheader("Temporal Evolution & Outbreak Dynamics (2023–2026)")
    
    # Epicurve analysis
    t_summary = TemporalStatsAnalyzer.analyze_epicurve(df_epicurve)
    df_epicurve_proc = t_summary["processed_df"]

    ep1, ep2, ep3 = st.columns(3)
    with ep1:
        st.metric("Total Surveillance Period Cases", f"{t_summary['total_cases_surveillance_period']:,}")
    with ep2:
        st.metric("National Epidemic Peak Month", t_summary["peak_month"])
    with ep3:
        st.metric("Peak Month Case Volume", f"{t_summary['peak_monthly_cases']:,} cases")

    fig_epi = ChartVisualizer.create_epicurve_chart(df_epicurve_proc, height=430)
    st.plotly_chart(fig_epi, use_container_width=True)

    col_stack, col_moran_trend = st.columns([6, 6])
    with col_stack:
        fig_area = ChartVisualizer.create_regional_stack_chart(df_epicurve_proc, height=420)
        st.plotly_chart(fig_area, use_container_width=True)

    with col_moran_trend:
        st.markdown("#### **Temporal Evolution of Spatial Autocorrelation**")
        df_moran_trend = TemporalStatsAnalyzer.track_temporal_moran_evolution(gdf_master, analyzer)
        
        fig_trend = px.line(
            df_moran_trend,
            x="Year",
            y="Moran's I",
            markers=True,
            title="Global Moran's I Trajectory Across Surveillance Years",
            text="Moran's I"
        )
        fig_trend.update_traces(textposition="top center", line_color="#3182CE", marker=dict(size=10, color="#E53E3E"))
        fig_trend.update_layout(height=360, yaxis_range=[0.0, 0.45], template="plotly_white")
        st.plotly_chart(fig_trend, use_container_width=True)
        st.caption("Strongest spatial autocorrelation occurred during the 2023/2024 outbreak surge before partial spatial diffusion.")


# =============================================================
# TAB 4: IMMUNISATION DEFICITS & RISK DRIVERS
# =============================================================
with tab_vaccine:
    st.subheader("Childhood MMR Vaccination Coverage & Disease Vulnerability")

    st.markdown("""
    <div class="warning-callout">
        <b>Public Health Herd Immunity Standard:</b> Measles has a high basic reproduction number ($R_0 \approx 12-18$). 
        To achieve herd immunity and prevent sustained community transmission, the <b>World Health Organization (WHO)</b> 
        recommends a minimum 2-dose MMR vaccination coverage of <b>95.0%</b> at age 5.
    </div>
    """, unsafe_allow_html=True)

    col_scatter, col_gap_table = st.columns([7, 5])

    with col_scatter:
        fig_vac = ChartVisualizer.create_mmr_vs_incidence_scatter(
            gdf_filtered,
            rate_col="cumulative_rate",
            mmr_col="mmr2_coverage_pct",
            height=540
        )
        st.plotly_chart(fig_vac, use_container_width=True)

    with col_gap_table:
        st.markdown("#### **Areas with Largest Immunisation Deficits**")
        st.caption("UTLAs falling furthest below the WHO 95% target, representing critical outbreak vulnerabilities.")
        df_vulnerable = gdf_filtered[["utla_name", "region_name", "mmr2_coverage_pct", "mmr_gap_to_target", "total_cases_2023_2026"]].sort_values("mmr_gap_to_target", ascending=False).head(10).reset_index(drop=True)
        df_vulnerable.columns = ["UTLA", "Region", "MMR2 Coverage (%)", "Immunity Gap (%)", "Total Cases"]
        
        st.dataframe(
            df_vulnerable.style.format({
                "MMR2 Coverage (%)": "{:.1f}%",
                "Immunity Gap (%)": "{:.1f}%",
                "Total Cases": "{:,}"
            }).background_gradient(subset=["Immunity Gap (%)"], cmap="OrRd"),
            use_container_width=True,
            height=460
        )


# =============================================================
# TAB 5: PUBLIC HEALTH INSIGHTS & METHODOLOGY
# =============================================================
with tab_insights:
    st.subheader("Public Health Evaluation, Limitations & Methodology")

    st.markdown("""
    ### 1. Key Epidemiological Findings
    1. **Outbreak Origin and Phased Spatial Transmission**:
       - The epidemic was triggered in late 2023 by an explosive local cluster in **Birmingham (West Midlands)**, where MMR2 coverage had fallen to ~75.2%.
       - In 2024, transmission disseminated nationally, heavily amplifying in **London** (accounting for 44.8% of national cases), with intense localized clusters in inner boroughs such as **Lambeth (178 cases)**, **Wandsworth (152 cases)**, and **Hackney (112 cases in 2024, rising to 133 in 2025)**.
    2. **Strong Empirical Association with Vaccine Coverage Gaps**:
       - There is an inverse statistical relationship between UTLA MMR2 vaccination rates and cumulative incidence ($p < 0.001$).
       - Clusters are concentrated where 2-dose coverage dips below 80%, well short of the WHO 95% threshold required to suppress measles transmission.
    3. **Confirmation of Significant Spatial Autocorrelation**:
       - Global Moran's $I$ is consistently positive and statistically significant ($I \approx 0.28 - 0.38, p < 0.001$), rejecting the null hypothesis of spatial randomness.
       - LISA and Getis-Ord $G_i^*$ analyses delineate persistent **High-High Hotspots** across the West Midlands and Inner London boroughs.

    ---

    ### 2. Methodological Strengths & Defenses for Interviews
    * **Standardized Rates over Raw Counts**: Raw counts conflate population size with disease burden. Calculating crude incidence per 100,000 avoids the classic "population map fallacy".
    * **Empirical Bayes Smoothing**: Mitigates rate volatility in small population authorities by shrinking unstable rates towards the national empirical prior.
    * **Sensitivity Testing on Spatial Topologies**: Cluster detection is verified under both Queen contiguity and $k$-nearest neighbours ($k=3, 5, 8$).
    * **Permutation-based Non-parametric Inference**: Monte Carlo simulation ($n=999$) ensures valid inference without relying on unrealistic asymptotic normality assumptions.

    ---

    ### 3. Surveillance Limitations
    * **Ecological Fallacy**: Aggregated UTLA-level statistics cannot establish individual-level causality.
    * **Modifiable Areal Unit Problem (MAUP)**: Zone boundaries (UTLA borders) are administrative rather than epidemiological; clustering patterns may vary if analyzed at Middle Layer Super Output Area (MSOA) scale.
    * **Ascertainment Bias**: Laboratory surveillance captures confirmed clinical presentations; milder or unreported community infections remain unobserved.
    """)
