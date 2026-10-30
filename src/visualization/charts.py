"""
src/visualization/charts.py
Epidemiological and spatial statistical chart generators using Plotly.
Includes Moran scatterplots, epidemic curves, regional area charts,
and vaccine coverage vs. disease incidence regressions.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal import weights
import plotly.express as px
import plotly.graph_objects as go

from src.config import LISA_COLORS, WHO_HERD_IMMUNITY_THRESHOLD_PCT


class ChartVisualizer:
    """
    Renders publication-ready epidemiological and statistical charts.
    """

    @staticmethod
    def create_moran_scatterplot(
        gdf: gpd.GeoDataFrame,
        variable: str,
        w: Optional[weights.W] = None,
        height: int = 550
    ) -> go.Figure:
        """
        Creates an interactive Anselin Moran Scatterplot.
        Horizontal axis: Standardized value (z).
        Vertical axis: Standardized spatial lag (Wz).
        The slope of the OLS line is Global Moran's I.
        """
        y = gdf[variable].values.astype(float)
        mean_y = np.mean(y)
        std_y = np.std(y) if np.std(y) > 0 else 1.0
        z = (y - mean_y) / std_y

        if w is None:
            w = weights.KNN.from_dataframe(gdf, k=5)
            w.transform = "R"

        wz = weights.lag_spatial(w, z)

        # OLS regression line
        slope, intercept = np.polyfit(z, wz, 1)
        x_line = np.linspace(np.min(z) - 0.2, np.max(z) + 0.2, 50)
        y_line = slope * x_line + intercept

        df_plot = pd.DataFrame({
            "utla_name": gdf["utla_name"],
            "region_name": gdf["region_name"],
            "standardized_val": z,
            "spatial_lag": wz,
            "lisa_cluster": gdf["lisa_cluster"] if "lisa_cluster" in gdf.columns else "All",
            "raw_value": y
        })

        fig = px.scatter(
            df_plot,
            x="standardized_val",
            y="spatial_lag",
            color="lisa_cluster",
            color_discrete_map=LISA_COLORS,
            hover_name="utla_name",
            hover_data={
                "region_name": True,
                "raw_value": ":.2f",
                "standardized_val": ":.2f",
                "spatial_lag": ":.2f"
            },
            title=f"Anselin Moran Scatterplot (Global Moran's I = {slope:.3f})"
        )

        # Add OLS regression line
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"Moran's I Slope ({slope:.3f})",
                line=dict(color="#1A202C", width=2.5, dash="dash")
            )
        )

        # Crosshairs at origin
        fig.add_hline(y=0, line=dict(color="#CBD5E0", width=1.5, dash="dot"))
        fig.add_vline(x=0, line=dict(color="#CBD5E0", width=1.5, dash="dot"))

        # Quadrant annotations
        fig.add_annotation(x=3.5, y=3.0, text="<b>High-High</b><br>(Hotspot)", showarrow=False, font=dict(color="#D73027", size=11))
        fig.add_annotation(x=-1.5, y=3.0, text="<b>Low-High</b><br>(Buffer)", showarrow=False, font=dict(color="#3182CE", size=11))
        fig.add_annotation(x=-1.5, y=-2.0, text="<b>Low-Low</b><br>(Coldspot)", showarrow=False, font=dict(color="#4575B4", size=11))
        fig.add_annotation(x=3.5, y=-2.0, text="<b>High-Low</b><br>(Outlier)", showarrow=False, font=dict(color="#DD6B20", size=11))

        fig.update_layout(
            xaxis_title="Standardized Incidence Rate (z)",
            yaxis_title="Spatial Lag of Standardized Rate (Wz)",
            height=height,
            template="plotly_white",
            legend=dict(title="LISA Category", orientation="h", y=-0.15, x=0.5, xanchor="center")
        )
        return fig

    @staticmethod
    def create_epicurve_chart(df_epicurve: pd.DataFrame, height: int = 420) -> go.Figure:
        """
        Creates an epidemiological curve bar chart with 3-month moving average overlay.
        """
        df = df_epicurve.copy().sort_values("date")
        if "rolling_3m_avg" not in df.columns:
            df["rolling_3m_avg"] = df["total_cases"].rolling(window=3, min_periods=1).mean()

        fig = go.Figure()

        # Monthly cases bar
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["total_cases"],
                name="Monthly Confirmed Cases",
                marker_color="#3182CE",
                opacity=0.85
            )
        )

        # Rolling average line
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["rolling_3m_avg"],
                name="3-Month Rolling Average",
                line=dict(color="#E53E3E", width=3)
            )
        )

        # Key epidemiological milestone annotations
        fig.add_vline(x="2023-11-01", line_dash="dash", line_color="#DD6B20", opacity=0.7)
        fig.add_annotation(
            x="2023-11-01", y=df["total_cases"].max() * 0.95,
            text="Nov 2023: Birmingham<br>Outbreak Ignition",
            showarrow=True, arrowhead=2, arrowcolor="#DD6B20", font=dict(size=10)
        )

        fig.add_vline(x="2024-04-01", line_dash="dash", line_color="#E53E3E", opacity=0.7)
        fig.add_annotation(
            x="2024-04-01", y=df["total_cases"].max() * 0.85,
            text="Spring 2024: Peak<br>London Epidemic Wave",
            showarrow=True, arrowhead=2, arrowcolor="#E53E3E", font=dict(size=10)
        )

        fig.update_layout(
            title="National Measles Epidemic Curve (England, 2023–2026)",
            xaxis_title="Surveillance Month",
            yaxis_title="Laboratory-Confirmed Cases",
            height=height,
            template="plotly_white",
            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center")
        )
        return fig

    @staticmethod
    def create_regional_stack_chart(df_epicurve: pd.DataFrame, height: int = 400) -> go.Figure:
        """
        Creates a stacked area chart showing dynamic regional contribution to the epidemic over time.
        """
        df = df_epicurve.copy().sort_values("date")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["cases_london"], mode="lines", name="London", stackgroup="one", fillcolor="rgba(229, 62, 62, 0.7)", line=dict(color="#E53E3E")))
        fig.add_trace(go.Scatter(x=df["date"], y=df["cases_west_midlands"], mode="lines", name="West Midlands", stackgroup="one", fillcolor="rgba(221, 107, 32, 0.7)", line=dict(color="#DD6B20")))
        fig.add_trace(go.Scatter(x=df["date"], y=df["cases_north_west"], mode="lines", name="North West", stackgroup="one", fillcolor="rgba(49, 130, 206, 0.7)", line=dict(color="#3182CE")))
        fig.add_trace(go.Scatter(x=df["date"], y=df["cases_other_regions"], mode="lines", name="Other English Regions", stackgroup="one", fillcolor="rgba(113, 128, 150, 0.5)", line=dict(color="#718096")))

        fig.update_layout(
            title="Regional Case Composition Across the Epidemic Surge",
            xaxis_title="Surveillance Month",
            yaxis_title="Monthly Cases",
            height=height,
            template="plotly_white",
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
        )
        return fig

    @staticmethod
    def create_mmr_vs_incidence_scatter(
        gdf: gpd.GeoDataFrame,
        rate_col: str = "cumulative_rate",
        mmr_col: str = "mmr2_coverage_pct",
        height: int = 480
    ) -> go.Figure:
        """
        Creates a scatterplot of MMR2 Coverage at 5 Years vs. Measles Incidence Rate.
        Features OLS trendline and reference vertical line at the WHO 95% target.
        """
        df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))

        fig = px.scatter(
            df,
            x=mmr_col,
            y=rate_col,
            size="population",
            color="region_name",
            hover_name="utla_name",
            hover_data={
                "population": ":,",
                rate_col: ":.2f",
                mmr_col: ":.1f%"
            },
            title="Vaccine Coverage vs. Measles Incidence Rate by UTLA",
            trendline="ols",
            trendline_color_override="#1A202C"
        )

        # WHO 95% Herd Immunity Line
        fig.add_vline(
            x=WHO_HERD_IMMUNITY_THRESHOLD_PCT,
            line_dash="dash",
            line_color="#38A169",
            line_width=2.5
        )
        fig.add_annotation(
            x=WHO_HERD_IMMUNITY_THRESHOLD_PCT,
            y=df[rate_col].max() * 0.95,
            text="WHO 95% Herd<br>Immunity Target",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#38A169",
            font=dict(color="#276749", size=10)
        )

        fig.update_layout(
            xaxis_title="MMR2 Vaccine Coverage at 5 Years (%)",
            yaxis_title="Measles Incidence Rate (per 100,000)",
            height=height,
            template="plotly_white",
            legend=dict(title="English Region", orientation="h", y=-0.2, x=0.5, xanchor="center")
        )
        return fig
