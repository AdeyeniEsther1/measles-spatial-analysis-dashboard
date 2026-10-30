"""
src/analysis/spatial_stats.py
Geospatial statistical analysis engine wrapping PySAL (libpysal and esda).
Implements Global Moran's I, Local Moran's I (LISA), and Getis-Ord Gi*
with Monte Carlo permutation inference and neighborhood sensitivity analysis.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal import weights
from esda.moran import Moran, Moran_Local
from esda.getisord import G_Local

from src.config import DEFAULT_SIGNIFICANCE_ALPHA, RANDOM_SEED


class SpatialStatsAnalyzer:
    """
    Computes spatial autocorrelation, local clustering, and hotspot statistics
    for measles incidence across England's administrative units.
    """

    def __init__(self, significance_alpha: float = DEFAULT_SIGNIFICANCE_ALPHA):
        self.alpha = significance_alpha

    @staticmethod
    def build_spatial_weights(
        gdf: gpd.GeoDataFrame, 
        method: str = "knn", 
        k: int = 5
    ) -> weights.W:
        """
        Constructs a row-standardized spatial weights matrix W.
        method: 'queen' (boundary contiguity) or 'knn' (k-nearest neighbours).
        For England administrative units, KNN(k=5) guarantees complete graph
        connectivity without disconnected island components (e.g. Isle of Wight).
        """
        if method.lower() == "queen":
            w = weights.Queen.from_dataframe(gdf, use_index=False)
            # Check for islands (islands have 0 neighbors in Queen)
            if len(w.islands) > 0:
                # Augment Queen weights with KNN for isolated polygons
                w_knn = weights.KNN.from_dataframe(gdf, k=2)
                for island_idx in w.islands:
                    w.neighbors[island_idx] = w_knn.neighbors[island_idx]
                w = weights.W(w.neighbors)
        else:
            w = weights.KNN.from_dataframe(gdf, k=k)

        w.transform = "R"  # Row standardisation: sum(w_ij) = 1
        return w

    def compute_global_moran(
        self, 
        gdf: gpd.GeoDataFrame, 
        variable: str, 
        w: Optional[weights.W] = None,
        permutations: int = 999
    ) -> Dict[str, Any]:
        """
        Calculates Global Moran's I statistic with analytical and
        Monte Carlo permutation-based significance testing.
        """
        if w is None:
            w = self.build_spatial_weights(gdf, method="knn", k=5)

        y = gdf[variable].values.astype(float)
        np.random.seed(RANDOM_SEED)
        mi = Moran(y, w, permutations=permutations)

        # Scientific interpretation
        if mi.p_sim < self.alpha:
            if mi.I > mi.EI:
                interpretation = "Statistically significant positive spatial clustering (Hotspots/Coldspots present)"
            else:
                interpretation = "Statistically significant spatial dispersion (Negative spatial autocorrelation)"
        else:
            interpretation = "Spatial pattern is not significantly different from spatial randomness"

        return {
            "moran_i": float(mi.I),
            "expected_i": float(mi.EI),
            "z_norm": float(mi.z_norm),
            "p_norm": float(mi.p_norm),
            "z_sim": float(mi.z_sim),
            "p_sim": float(mi.p_sim),
            "permutations": permutations,
            "is_significant": bool(mi.p_sim < self.alpha),
            "interpretation": interpretation
        }

    def compute_local_moran(
        self, 
        gdf: gpd.GeoDataFrame, 
        variable: str, 
        w: Optional[weights.W] = None,
        permutations: int = 999
    ) -> gpd.GeoDataFrame:
        """
        Calculates Local Indicators of Spatial Association (LISA).
        Classifies polygons into High-High, Low-Low, High-Low, Low-High,
        or Not Significant at the specified significance level alpha.
        """
        if w is None:
            w = self.build_spatial_weights(gdf, method="knn", k=5)

        y = gdf[variable].values.astype(float)
        np.random.seed(RANDOM_SEED)
        lm = Moran_Local(y, w, permutations=permutations, seed=RANDOM_SEED, alternative="two-sided")

        gdf_out = gdf.copy()
        gdf_out["lisa_I"] = lm.Is
        gdf_out["lisa_p"] = lm.p_sim
        gdf_out["lisa_z"] = lm.z_sim
        gdf_out["spatial_lag"] = weights.lag_spatial(w, y)

        # Standard quadrant labels
        quadrant_map = {
            1: "High-High (Hotspot)",
            2: "Low-High (Buffer)",
            3: "Low-Low (Coldspot)",
            4: "High-Low (Outlier)"
        }

        # Apply significance mask
        sig_mask = lm.p_sim < self.alpha
        clusters = []
        for q, is_sig in zip(lm.q, sig_mask):
            if is_sig:
                clusters.append(quadrant_map.get(q, "Not Significant"))
            else:
                clusters.append("Not Significant")

        gdf_out["lisa_cluster"] = clusters
        return gdf_out

    def compute_getis_ord_gi(
        self, 
        gdf: gpd.GeoDataFrame, 
        variable: str, 
        w: Optional[weights.W] = None,
        permutations: int = 999
    ) -> gpd.GeoDataFrame:
        """
        Calculates Getis-Ord Gi* (star=True includes focal polygon).
        Outputs standard z-scores and 7-tier confidence classifications.
        """
        if w is None:
            w = self.build_spatial_weights(gdf, method="knn", k=5)

        y = gdf[variable].values.astype(float)
        np.random.seed(RANDOM_SEED)
        
        # G_Local with star=1.0 computes Gi* with explicit self-weight
        gi = G_Local(y, w, transform="R", star=1.0, permutations=permutations, seed=RANDOM_SEED, alternative="two-sided")

        gdf_out = gdf.copy()
        gdf_out["gi_z"] = gi.Zs
        gdf_out["gi_p"] = gi.p_sim

        # Classification into confidence bins
        categories = []
        for z, p in zip(gi.Zs, gi.p_sim):
            if p <= 0.01 and z > 2.58:
                categories.append("Hot Spot (99% Conf.)")
            elif p <= 0.05 and z > 1.96:
                categories.append("Hot Spot (95% Conf.)")
            elif p <= 0.10 and z > 1.65:
                categories.append("Hot Spot (90% Conf.)")
            elif p <= 0.01 and z < -2.58:
                categories.append("Cold Spot (99% Conf.)")
            elif p <= 0.05 and z < -1.96:
                categories.append("Cold Spot (95% Conf.)")
            elif p <= 0.10 and z < -1.65:
                categories.append("Cold Spot (90% Conf.)")
            else:
                categories.append("Not Significant")

        gdf_out["gi_category"] = categories
        return gdf_out

    def run_spatial_sensitivity_analysis(
        self, 
        gdf: gpd.GeoDataFrame, 
        variable: str
    ) -> pd.DataFrame:
        """
        Evaluates the sensitivity of Global Moran's I across multiple
        neighborhood specifications (Queen contiguity vs KNN k=3, 5, 8).
        """
        specs = [
            ("Queen Contiguity", "queen", None),
            ("KNN (k=3)", "knn", 3),
            ("KNN (k=5)", "knn", 5),
            ("KNN (k=8)", "knn", 8),
        ]
        results = []
        for label, method, k in specs:
            w = self.build_spatial_weights(gdf, method=method, k=k or 5)
            stats = self.compute_global_moran(gdf, variable, w=w, permutations=499)
            results.append({
                "Neighborhood Specification": label,
                "Moran's I": round(stats["moran_i"], 4),
                "z-Score": round(stats["z_sim"], 3),
                "Pseudo p-value": stats["p_sim"],
                "Clustering Detected": "Yes" if stats["is_significant"] and stats["moran_i"] > 0 else "No"
            })
        return pd.DataFrame(results)
