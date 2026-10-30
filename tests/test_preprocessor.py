"""
tests/test_preprocessor.py
Unit and integration tests for data preprocessing and rate standardization.
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd

from src.data.preprocessor import DataPreprocessor
from src.data.loader import DataLoader


def test_calculate_crude_rates():
    df = pd.DataFrame({
        "cases": [10, 50],
        "pop": [100000, 250000]
    })
    out = DataPreprocessor.calculate_crude_rates(df, "cases", "pop", "rate")
    assert np.isclose(out["rate"].iloc[0], 10.0)
    assert np.isclose(out["rate"].iloc[1], 20.0)


def test_empirical_bayes_shrinkage():
    # Large area with standard rate vs small area with extreme rate
    df = pd.DataFrame({
        "cases": [1000, 5, 2],
        "pop": [1000000, 5000, 100000]
    })
    df = DataPreprocessor.calculate_crude_rates(df, "cases", "pop", "crude_rate")
    df = DataPreprocessor.calculate_empirical_bayes_rates(df, "cases", "pop", "eb_rate")

    # The small area (pop=5000, cases=5, crude=100) should be shrunk towards the national mean
    assert df["eb_rate"].iloc[1] < df["crude_rate"].iloc[1]
    assert df["eb_rate"].iloc[0] == pytest.approx(df["crude_rate"].iloc[0], rel=0.1)


def test_build_analytical_dataset_integration():
    loader = DataLoader()
    preprocessor = DataPreprocessor(loader)
    gdf = preprocessor.build_analytical_dataset(save_processed=False)

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) > 100
    assert "rate_2024" in gdf.columns
    assert "eb_rate_2024" in gdf.columns
    assert "mmr_gap_to_target" in gdf.columns
    assert gdf.crs.to_string() == "EPSG:4326"
    assert not gdf.geometry.isnull().any()
