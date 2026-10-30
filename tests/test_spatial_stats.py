"""
tests/test_spatial_stats.py
Unit tests for spatial autocorrelation statistics (Moran's I, LISA, Getis-Ord Gi*).
"""

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

from src.data.preprocessor import DataPreprocessor
from src.analysis.spatial_stats import SpatialStatsAnalyzer


@pytest.fixture(scope="module")
def analytical_gdf():
    preprocessor = DataPreprocessor()
    return preprocessor.build_analytical_dataset(save_processed=False)


def test_spatial_weights_construction(analytical_gdf):
    analyzer = SpatialStatsAnalyzer()
    w_knn = analyzer.build_spatial_weights(analytical_gdf, method="knn", k=5)
    assert w_knn.n == len(analytical_gdf)
    assert len(w_knn.islands) == 0

    w_queen = analyzer.build_spatial_weights(analytical_gdf, method="queen")
    assert w_queen.n == len(analytical_gdf)
    # Islands should be reconnected
    assert len(w_queen.islands) == 0


def test_global_moran_computation(analytical_gdf):
    analyzer = SpatialStatsAnalyzer()
    res = analyzer.compute_global_moran(analytical_gdf, "rate_2024", permutations=199)

    assert -1.0 <= res["moran_i"] <= 1.0
    assert 0.0 <= res["p_sim"] <= 1.0
    assert "interpretation" in res
    # Real measles epidemic data exhibits significant positive spatial autocorrelation
    assert res["moran_i"] > 0
    assert res["is_significant"] is True


def test_local_moran_lisa_quadrants(analytical_gdf):
    analyzer = SpatialStatsAnalyzer()
    gdf_lisa = analyzer.compute_local_moran(analytical_gdf, "rate_2024", permutations=199)

    assert "lisa_cluster" in gdf_lisa.columns
    assert "lisa_p" in gdf_lisa.columns
    assert "spatial_lag" in gdf_lisa.columns

    valid_clusters = {
        "High-High (Hotspot)",
        "Low-Low (Coldspot)",
        "High-Low (Outlier)",
        "Low-High (Buffer)",
        "Not Significant"
    }
    assert set(gdf_lisa["lisa_cluster"].unique()).issubset(valid_clusters)

    # Birmingham or Hackney should be detected as Hotspot or high outlier
    hotspot_areas = gdf_lisa[gdf_lisa["lisa_cluster"] == "High-High (Hotspot)"]["utla_name"].tolist()
    assert len(hotspot_areas) > 0


def test_getis_ord_gi_star(analytical_gdf):
    analyzer = SpatialStatsAnalyzer()
    gdf_gi = analyzer.compute_getis_ord_gi(analytical_gdf, "rate_2024", permutations=199)

    assert "gi_z" in gdf_gi.columns
    assert "gi_category" in gdf_gi.columns
    assert not gdf_gi["gi_z"].isnull().any()


def test_spatial_sensitivity_analysis(analytical_gdf):
    analyzer = SpatialStatsAnalyzer()
    df_sens = analyzer.run_spatial_sensitivity_analysis(analytical_gdf, "rate_2024")

    assert len(df_sens) == 4
    assert "Neighborhood Specification" in df_sens.columns
    assert "Moran's I" in df_sens.columns
