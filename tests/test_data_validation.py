"""
tests/test_data_validation.py
Unit tests for data validation, schema enforcement, and CRS checks.
"""

import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

from src.data.validator import DataValidator, ValidationError


def test_validate_utla_cases_success():
    valid_df = pd.DataFrame({
        "utla_code": ["E08000025", "E09000012"],
        "utla_name": ["Birmingham", "Hackney"],
        "cases_2023": [118, 24],
        "cases_2024": [364, 112],
        "cases_2025": [52, 133],
        "cases_2026": [58, 45]
    })
    assert DataValidator.validate_utla_cases(valid_df) is True


def test_validate_utla_cases_missing_column():
    invalid_df = pd.DataFrame({
        "utla_code": ["E08000025"],
        "utla_name": ["Birmingham"]
    })
    with pytest.raises(ValidationError, match="missing columns"):
        DataValidator.validate_utla_cases(invalid_df)


def test_validate_utla_cases_negative_count():
    invalid_df = pd.DataFrame({
        "utla_code": ["E08000025"],
        "utla_name": ["Birmingham"],
        "cases_2023": [-5],
        "cases_2024": [10],
        "cases_2025": [0],
        "cases_2026": [2]
    })
    with pytest.raises(ValidationError, match="Negative case count"):
        DataValidator.validate_utla_cases(invalid_df)


def test_validate_populations_positive():
    valid_df = pd.DataFrame({
        "utla_code": ["E08000025", "E09000012"],
        "population": [1144900, 280900]
    })
    assert DataValidator.validate_populations(valid_df) is True

    invalid_df = pd.DataFrame({
        "utla_code": ["E08000025"],
        "population": [0]
    })
    with pytest.raises(ValidationError, match="strictly positive"):
        DataValidator.validate_populations(invalid_df)


def test_validate_mmr_coverage_range():
    valid_df = pd.DataFrame({"mmr2_coverage_pct": [85.5, 95.0, 72.1]})
    assert DataValidator.validate_mmr_coverage(valid_df) is True

    invalid_df = pd.DataFrame({"mmr2_coverage_pct": [105.0, 60.0]})
    with pytest.raises(ValidationError, match="between 0% and 100%"):
        DataValidator.validate_mmr_coverage(invalid_df)


def test_validate_geodataframe():
    p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf_valid = gpd.GeoDataFrame({"id": [1]}, geometry=[p1], crs="EPSG:4326")
    assert DataValidator.validate_geodataframe(gdf_valid, expected_crs="EPSG:4326") is True

    # Test invalid CRS
    with pytest.raises(ValidationError, match="Expected CRS"):
        DataValidator.validate_geodataframe(gdf_valid, expected_crs="EPSG:27700")
