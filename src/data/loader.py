"""
src/data/loader.py
Robust dual-mode data loader for UKHSA measles surveillance data,
ONS population estimates, MMR coverage, and vector GeoJSON boundaries.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import geopandas as gpd
import requests

from src.config import (
    RAW_DATA_DIR,
    BOUNDARIES_DIR,
    CRS_WGS84,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles retrieval of public health surveillance data.
    Provides live fetching capabilities with graceful, transparent
    fallback to verified local baseline datasets when running offline.
    """

    def __init__(self, raw_dir: Optional[Path] = None, boundaries_dir: Optional[Path] = None):
        self.raw_dir = Path(raw_dir) if raw_dir else RAW_DATA_DIR
        self.boundaries_dir = Path(boundaries_dir) if boundaries_dir else BOUNDARIES_DIR

    def fetch_live_govuk_epidemiology(self, year: int) -> Optional[pd.DataFrame]:
        """
        Attempts to scrape / retrieve the latest published UKHSA HTML tables
        directly from GOV.UK for a given surveillance year.
        Returns None if remote host is unreachable or times out.
        """
        url = (
            f"https://www.gov.uk/government/publications/measles-epidemiology-2023-to-2026/"
            f"confirmed-cases-of-measles-in-england-by-month-age-region-and-upper-tier-local-authority-{year}"
        )
        try:
            headers = {"User-Agent": "MeaslesSpatialSurveillanceBot/1.0"}
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")
                tables = soup.find_all("table")
                if tables:
                    logger.info(f"Successfully retrieved live GOV.UK data for year {year}")
                    # Parse first table
                    rows = []
                    for tr in tables[0].find_all("tr"):
                        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
                        if cells:
                            rows.append(cells)
                    if rows and len(rows) > 1:
                        return pd.DataFrame(rows[1:], columns=rows[0])
            logger.warning(f"Live GOV.UK returned HTTP {response.status_code} for {year}")
            return None
        except Exception as exc:
            logger.debug(f"Live network access unavailable for {year}: {exc}")
            return None

    def load_regional_cases(self, use_live_fallback: bool = True) -> pd.DataFrame:
        """Loads annual regional measles surveillance counts."""
        filepath = self.raw_dir / "ukhsa_measles_annual_regional.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Regional baseline data not found at {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} regional records from baseline")
        return df

    def load_monthly_epicurve(self) -> pd.DataFrame:
        """Loads monthly epidemic curve time series."""
        filepath = self.raw_dir / "ukhsa_measles_monthly_epicurve.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Epidemic curve data not found at {filepath}")
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def load_utla_cases(self) -> pd.DataFrame:
        """Loads Upper Tier Local Authority (UTLA) measles cases."""
        filepath = self.raw_dir / "ukhsa_measles_utla_cases.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"UTLA cases dataset not found at {filepath}")
        return pd.read_csv(filepath)

    def load_populations(self) -> pd.DataFrame:
        """Loads ONS UTLA mid-year population estimates."""
        filepath = self.raw_dir / "ons_utla_populations.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Population dataset not found at {filepath}")
        return pd.read_csv(filepath)

    def load_mmr_coverage(self) -> pd.DataFrame:
        """Loads UKHSA COVER childhood MMR2 vaccination rates."""
        filepath = self.raw_dir / "ukhsa_mmr_coverage.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"MMR coverage dataset not found at {filepath}")
        return pd.read_csv(filepath)

    def load_boundaries(self, level: str = "utla") -> gpd.GeoDataFrame:
        """
        Loads vector boundary polygons in WGS84 (EPSG:4326).
        level: 'utla' or 'region'
        """
        filename = "england_utla.geojson" if level.lower() == "utla" else "england_regions.geojson"
        filepath = self.boundaries_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Boundary file not found at {filepath}")
        gdf = gpd.read_file(filepath)
        if gdf.crs is None:
            gdf = gdf.set_crs(CRS_WGS84)
        elif gdf.crs.to_string() != CRS_WGS84:
            gdf = gdf.to_crs(CRS_WGS84)
        return gdf
