"""
src/analysis/temporal_stats.py
Temporal surveillance analysis module.
Deconstructs epidemic curves, calculates growth rates and moving averages,
and tracks the temporal evolution of spatial clustering.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


class TemporalStatsAnalyzer:
    """
    Computes time-series and spatio-temporal dynamics for measles cases
    across English surveillance periods.
    """

    @staticmethod
    def analyze_epicurve(df_epicurve: pd.DataFrame) -> Dict[str, Any]:
        """
        Extracts peak month, maximum monthly case volume, total cases,
        and rolling averages from the epidemic curve time-series.
        """
        df = df_epicurve.copy().sort_values("date").reset_index(drop=True)
        
        # 3-month rolling average
        df["rolling_3m_avg"] = df["total_cases"].rolling(window=3, min_periods=1).mean()
        
        # Month-on-month percentage change
        df["mom_growth_pct"] = df["total_cases"].pct_change() * 100.0
        df["mom_growth_pct"] = df["mom_growth_pct"].fillna(0.0)

        # Detect peak
        peak_idx = df["total_cases"].idxmax()
        peak_row = df.iloc[peak_idx]

        return {
            "total_cases_surveillance_period": int(df["total_cases"].sum()),
            "peak_month": str(peak_row["date"].strftime("%B %Y")),
            "peak_monthly_cases": int(peak_row["total_cases"]),
            "processed_df": df
        }

    @staticmethod
    def compute_regional_share_by_year(df_regional: pd.DataFrame) -> pd.DataFrame:
        """
        Computes regional percentage contribution to the national epidemic
        across each surveillance year (2023 to 2026).
        """
        years = [2023, 2024, 2025, 2026]
        df_out = pd.DataFrame()
        df_out["region_name"] = df_regional["region_name"]

        for y in years:
            total_y = df_regional[f"cases_{y}"].sum()
            if total_y > 0:
                df_out[f"share_{y}_pct"] = (df_regional[f"cases_{y}"] / total_y) * 100.0
            else:
                df_out[f"share_{y}_pct"] = 0.0

        return df_out

    @staticmethod
    def track_temporal_moran_evolution(
        gdf: Any, 
        analyzer: Any, 
        years: List[int] = [2023, 2024, 2025, 2026]
    ) -> pd.DataFrame:
        """
        Tracks Global Moran's I across years to evaluate how spatial clustering
        strengthened or diffused over the epidemic trajectory.
        """
        records = []
        for y in years:
            col = f"rate_{y}"
            if col in gdf.columns:
                res = analyzer.compute_global_moran(gdf, col)
                records.append({
                    "Year": y,
                    "Moran's I": round(res["moran_i"], 4),
                    "Expected I": round(res["expected_i"], 4),
                    "z-score": round(res["z_sim"], 3),
                    "p-value": res["p_sim"],
                    "Significant Clustering": "Yes" if res["is_significant"] and res["moran_i"] > 0 else "No"
                })
        return pd.DataFrame(records)
