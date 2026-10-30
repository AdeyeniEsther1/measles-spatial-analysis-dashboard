"""
src/visualization/maps.py
Interactive geospatial map visualizers using Plotly Express and Graph Objects.
Renders choropleths, Local Moran's I (LISA) cluster maps, and Getis-Ord Gi* hotspot maps.
"""

import json
from typing import Dict, Any, Optional
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

from src.config import LISA_COLORS, GI_STAR_COLORS, INCIDENCE_COLORSCALE


class MapVisualizer:
    """
    Renders high-quality interactive geospatial maps with custom
    hover annotations, unified palettes, and responsive layouts.
    """

    @staticmethod
    def _prepare_geojson(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Converts GeoDataFrame to JSON dict with WGS84 coordinates."""
        if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        return json.loads(gdf.to_json())

    @classmethod
    def create_choropleth(
        cls,
        gdf: gpd.GeoDataFrame,
        color_col: str,
        title: str,
        color_continuous_scale: str = INCIDENCE_COLORSCALE,
        hover_name: str = "utla_name",
        hover_data: Optional[Dict[str, Any]] = None,
        height: int = 650
    ) -> go.Figure:
        """
        Creates an interactive continuous choropleth map.
        """
        geojson_data = cls._prepare_geojson(gdf)
        
        default_hover = {
            "utla_name": True,
            "region_name": True,
            "population": ":,",
            color_col: ":.2f"
        }
        if hover_data:
            default_hover.update(hover_data)

        fig = px.choropleth_map(
            gdf,
            geojson=geojson_data,
            locations="utla_code",
            featureidkey="properties.utla_code",
            color=color_col,
            color_continuous_scale=color_continuous_scale,
            map_style="carto-positron",
            center={"lat": 52.8, "lon": -1.5},
            zoom=5.6,
            opacity=0.85,
            hover_name=hover_name,
            hover_data=default_hover,
            title=title
        )

        fig.update_layout(
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            height=height,
            font=dict(family="Arial, sans-serif", size=12),
            coloraxis_colorbar=dict(
                title=dict(text=color_col.replace("_", " ").title(), side="right"),
                thicknessmode="pixels",
                thickness=15,
                lenmode="fraction",
                len=0.75
            )
        )
        return fig

    @classmethod
    def create_lisa_map(
        cls,
        gdf: gpd.GeoDataFrame,
        cluster_col: str = "lisa_cluster",
        title: str = "Local Moran's I (LISA) Spatial Clusters",
        height: int = 650
    ) -> go.Figure:
        """
        Creates a categorical choropleth map for LISA quadrants.
        """
        geojson_data = cls._prepare_geojson(gdf)

        fig = px.choropleth_map(
            gdf,
            geojson=geojson_data,
            locations="utla_code",
            featureidkey="properties.utla_code",
            color=cluster_col,
            color_discrete_map=LISA_COLORS,
            category_orders={cluster_col: list(LISA_COLORS.keys())},
            map_style="carto-positron",
            center={"lat": 52.8, "lon": -1.5},
            zoom=5.6,
            opacity=0.85,
            hover_name="utla_name",
            hover_data={
                "region_name": True,
                cluster_col: True,
                "population": ":,"
            },
            title=title
        )

        fig.update_layout(
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            height=height,
            font=dict(family="Arial, sans-serif", size=12),
            legend=dict(
                title="LISA Cluster Type",
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        )
        return fig

    @classmethod
    def create_gi_star_map(
        cls,
        gdf: gpd.GeoDataFrame,
        category_col: str = "gi_category",
        title: str = "Getis-Ord Gi* Hotspot Analysis",
        height: int = 650
    ) -> go.Figure:
        """
        Creates a categorical choropleth map for Getis-Ord Gi* confidence tiers.
        """
        geojson_data = cls._prepare_geojson(gdf)

        fig = px.choropleth_map(
            gdf,
            geojson=geojson_data,
            locations="utla_code",
            featureidkey="properties.utla_code",
            color=category_col,
            color_discrete_map=GI_STAR_COLORS,
            category_orders={category_col: list(GI_STAR_COLORS.keys())},
            map_style="carto-positron",
            center={"lat": 52.8, "lon": -1.5},
            zoom=5.6,
            opacity=0.85,
            hover_name="utla_name",
            hover_data={
                "region_name": True,
                category_col: True,
                "gi_z": ":.2f",
                "gi_p": ":.3f"
            },
            title=title
        )

        fig.update_layout(
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            height=height,
            font=dict(family="Arial, sans-serif", size=12),
            legend=dict(
                title="Confidence Level",
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            )
        )
        return fig
