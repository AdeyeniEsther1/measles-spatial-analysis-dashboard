"""
src/data/preprocessor.py
Data preprocessing, feature engineering, and rate standardization module.
Combines tabular surveillance data with vector geometries and implements
Empirical Bayes rate smoothing.
"""

from typing import Optional
import numpy as np
import pandas as pd
import geopandas as gpd

from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.config import PROCESSED_DATA_DIR, WHO_HERD_IMMUNITY_THRESHOLD_PCT


class DataPreprocessor:
    """
    Transforms raw surveillance, demographic, and spatial data
    into a standardized analytical GeoDataFrame.
    """

    def __init__(self, loader: Optional[DataLoader] = None):
        self.loader = loader or DataLoader()
        self.validator = DataValidator()

    @staticmethod
    def calculate_crude_rates(df: pd.DataFrame, case_col: str, pop_col: str, rate_col: str) -> pd.DataFrame:
        """Calculates crude incidence rate per 100,000 population."""
        df = df.copy()
        df[rate_col] = (df[case_col] / df[pop_col]) * 100000.0
        return df

    @staticmethod
    def calculate_empirical_bayes_rates(
        df: pd.DataFrame, 
        case_col: str, 
        pop_col: str, 
        eb_rate_col: str
    ) -> pd.DataFrame:
        """
        Calculates Empirical Bayes (Clayton-Kaldor) smoothed incidence rates per 100,000.
        Shrinks volatile crude rates in small populations toward the national mean rate.
        """
        df = df.copy()
        cases = df[case_col].values.astype(float)
        pops = df[pop_col].values.astype(float)

        total_cases = np.sum(cases)
        total_pop = np.sum(pops)

        if total_pop == 0:
            df[eb_rate_col] = 0.0
            return df

        # Overall national incidence rate (per person)
        global_rate = total_cases / total_pop
        crude_rates = cases / pops

        # Mean population
        mean_pop = np.mean(pops)

        # Variance of raw rates
        var_crude = np.var(crude_rates)
        # Prior variance (ensuring non-negative)
        prior_var = max(0.0, var_crude - (global_rate / mean_pop))

        # Shrinkage weights w_i in [0, 1]
        denom = prior_var + (global_rate / pops)
        denom[denom == 0] = 1e-9
        weights = prior_var / denom
        weights = np.clip(weights, 0.0, 1.0)

        # Smoothed rate per person
        smoothed_rates = weights * crude_rates + (1.0 - weights) * global_rate

        # Convert to rate per 100,000
        df[eb_rate_col] = smoothed_rates * 100000.0
        return df

    def build_analytical_dataset(self, save_processed: bool = True) -> gpd.GeoDataFrame:
        """
        Orchestrates ingestion, validation, joins, rate calculations,
        and spatial boundary binding.
        """
        # 1. Load data
        df_cases = self.loader.load_utla_cases()
        df_pops = self.loader.load_populations()
        df_mmr = self.loader.load_mmr_coverage()
        gdf_boundaries = self.loader.load_boundaries(level="utla")

        # 2. Validate
        self.validator.validate_utla_cases(df_cases)
        self.validator.validate_populations(df_pops)
        self.validator.validate_mmr_coverage(df_mmr)
        self.validator.validate_geodataframe(gdf_boundaries)

        # 3. Join tabular datasets
        df_merged = df_cases.merge(
            df_pops[["utla_code", "population"]], on="utla_code", how="left"
        ).merge(
            df_mmr[["utla_code", "mmr2_coverage_pct"]], on="utla_code", how="left"
        )

        # 4. Standardise Rates
        years = [2023, 2024, 2025, 2026]
        for y in years:
            c_col = f"cases_{y}"
            r_col = f"rate_{y}"
            eb_col = f"eb_rate_{y}"
            df_merged = self.calculate_crude_rates(df_merged, c_col, "population", r_col)
            df_merged = self.calculate_empirical_bayes_rates(df_merged, c_col, "population", eb_col)

        # Cumulative rates
        df_merged = self.calculate_crude_rates(
            df_merged, "total_cases_2023_2026", "population", "cumulative_rate"
        )
        df_merged = self.calculate_empirical_bayes_rates(
            df_merged, "total_cases_2023_2026", "population", "eb_cumulative_rate"
        )

        # MMR Immunity Gap
        df_merged["mmr_gap_to_target"] = np.maximum(
            0.0, WHO_HERD_IMMUNITY_THRESHOLD_PCT - df_merged["mmr2_coverage_pct"]
        )

        # 5. Join to GeoDataFrame
        gdf_analytical = gdf_boundaries[["utla_code", "geometry"]].merge(
            df_merged, on="utla_code", how="inner"
        )

        gdf_analytical = gpd.GeoDataFrame(gdf_analytical, geometry="geometry", crs=gdf_boundaries.crs)

        # 6. Save processed dataset
        if save_processed:
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            output_parquet = PROCESSED_DATA_DIR / "measles_analytical_master.parquet"
            # Pyogrio / geopandas parquet export
            gdf_analytical.to_parquet(output_parquet)

        return gdf_analytical
