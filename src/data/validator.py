"""
src/data/validator.py
Data integrity and geospatial validation module.
Enforces schema constraints, range assertions, and CRS validity.
"""

from typing import List, Dict, Any
import pandas as pd
import geopandas as gpd


class ValidationError(Exception):
    """Raised when data integrity validation fails."""
    pass


class DataValidator:
    """
    Validates dataframes and geodataframes for completeness,
    schema adherence, and epidemiological plausibility.
    """

    @staticmethod
    def validate_utla_cases(df: pd.DataFrame) -> bool:
        """Validates UTLA cases schema and non-negativity."""
        required_cols = ["utla_code", "utla_name", "cases_2023", "cases_2024", "cases_2025", "cases_2026"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValidationError(f"UTLA cases dataset missing columns: {missing}")

        case_cols = ["cases_2023", "cases_2024", "cases_2025", "cases_2026"]
        for col in case_cols:
            if (df[col] < 0).any():
                raise ValidationError(f"Negative case count detected in column '{col}'")
            if df[col].isnull().any():
                raise ValidationError(f"Null values detected in column '{col}'")

        if df["utla_code"].duplicated().any():
            dups = df[df["utla_code"].duplicated()]["utla_code"].tolist()
            raise ValidationError(f"Duplicate UTLA codes found: {dups}")

        return True

    @staticmethod
    def validate_populations(df: pd.DataFrame) -> bool:
        """Validates population estimates (must be strictly positive)."""
        required_cols = ["utla_code", "population"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValidationError(f"Population dataset missing columns: {missing}")

        if (df["population"] <= 0).any():
            raise ValidationError("Population counts must be strictly positive (> 0)")

        if df["population"].isnull().any():
            raise ValidationError("Null population values detected")

        return True

    @staticmethod
    def validate_mmr_coverage(df: pd.DataFrame) -> bool:
        """Validates MMR coverage percentages (0 to 100%)."""
        if "mmr2_coverage_pct" not in df.columns:
            raise ValidationError("Missing 'mmr2_coverage_pct' column")

        if ((df["mmr2_coverage_pct"] < 0) | (df["mmr2_coverage_pct"] > 100)).any():
            raise ValidationError("MMR2 coverage percentages must be between 0% and 100%")

        return True

    @staticmethod
    def validate_geodataframe(gdf: gpd.GeoDataFrame, expected_crs: str = "EPSG:4326") -> bool:
        """Validates geometry validity, non-emptiness, and CRS."""
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValidationError("Object is not a GeoPandas GeoDataFrame")

        if "geometry" not in gdf.columns:
            raise ValidationError("GeoDataFrame has no 'geometry' column")

        if gdf.geometry.isnull().any() or gdf.geometry.is_empty.any():
            raise ValidationError("GeoDataFrame contains null or empty geometries")

        if not gdf.geometry.is_valid.all():
            invalid_idx = gdf[~gdf.geometry.is_valid].index.tolist()
            raise ValidationError(f"Invalid polygon geometries detected at indices: {invalid_idx}")

        if gdf.crs is None:
            raise ValidationError("GeoDataFrame CRS is undefined")

        # Compare authority and code
        gdf_crs_str = gdf.crs.to_string().upper()
        if expected_crs.upper() not in gdf_crs_str:
            raise ValidationError(f"Expected CRS '{expected_crs}', but got '{gdf_crs_str}'")

        return True
