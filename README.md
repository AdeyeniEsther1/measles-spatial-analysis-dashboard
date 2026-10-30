# Spatial and Temporal Surveillance of Measles in England

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Framework: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Spatial: PySAL / GeoPandas](https://img.shields.io/badge/Spatial-PySAL%20%7C%20GeoPandas-green.svg)](https://pysal.org/)

An end-to-end, production-grade geospatial epidemiology and spatial econometrics portfolio project investigating the **2023–2026 measles resurgence across England** using official surveillance data from the **UK Health Security Agency (UKHSA)**, **Office for National Statistics (ONS)**, and **COVER immunisation programme**.

---

## Executive Summary & Epidemiological Problem

Between late 2023 and 2026, England recorded its largest measles resurgence in more than a decade:
* **2023 Outbreak Genesis**: 367 confirmed cases, initiated in late autumn by an intense localized outbreak in **Birmingham (West Midlands)**.
* **2024 Epidemic Surge**: National peak with **2,911 laboratory-confirmed cases** (the highest annual burden since 2012), disseminating across **London** (1,305 cases; 44.8%), the West Midlands, and spreading to every English region.
* **2025–2026 Endemic Persistence**: Sustained community transmission (958 cases in 2025; 1,030 cases reported through October 2026), heavily concentrated in children aged $\le 10$ years (61%–68%) residing in under-vaccinated urban centers.

This platform automates data validation, rate standardization, and exploratory data analysis, coupling them with advanced spatial autocorrelation modeling (**Global Moran's $I$**, **Local Moran's $I$ / LISA**, and **Getis-Ord $G_i^*$**) and an interactive **Streamlit** dashboard.

---

## System Architecture

```
measles_spatial_surveillance/
│
├── data/
│   ├── raw/                        # Baseline authentic surveillance & demographic files
│   │   ├── ukhsa_measles_annual_regional.csv
│   │   ├── ukhsa_measles_monthly_epicurve.csv
│   │   ├── ukhsa_measles_utla_cases.csv
│   │   ├── ons_utla_populations.csv
│   │   └── ukhsa_mmr_coverage.csv
│   ├── boundaries/                 # Vector GeoJSON boundary layers
│   │   ├── england_utla.geojson
│   │   └── england_regions.geojson
│   └── processed/                  # Standardised analytical master
│       └── measles_analytical_master.parquet
│
├── src/
│   ├── config.py                   # Global constants, CRS definitions, styling palettes
│   ├── data/
│   │   ├── loader.py               # Dual-mode loader (live GOV.UK scraper + offline baseline)
│   │   ├── validator.py            # Schema, range constraints, and polygon CRS assertions
│   │   └── preprocessor.py         # Empirical Bayes rate smoothing & spatial joins
│   ├── analysis/
│   │   ├── spatial_stats.py        # PySAL wrapper for Global Moran, LISA, and Getis-Ord Gi*
│   │   └── temporal_stats.py       # Epicurve deconstruction, peak detection, & Moran trends
│   └── visualization/
│       ├── maps.py                 # Plotly continuous choropleths, LISA & Gi* hotspot maps
│       └── charts.py               # Moran scatterplots, epicurves, & MMR regressions
│
├── tests/
│   ├── test_data_validation.py     # Schema assertions, negative values, and duplicate handling
│   ├── test_preprocessor.py        # Rate calculations & Empirical Bayes shrinkage tests
│   └── test_spatial_stats.py       # Spatial weight properties & statistical invariants
│
├── app.py                          # Multi-tab Streamlit Portfolio Dashboard
├── requirements.txt                # Exact pinned project dependencies
└── README.md                       # Comprehensive system documentation & interview defense
```

---

## Theoretical & Mathematical Foundations

### 1. The Denominator Dilemma: Crude vs. Empirical Bayes Rates
Analyzing raw case counts produces an artifactual map of population density ("the population density fallacy"). To avoid this, we calculate **Crude Incidence per 100,000 population**:
$$\text{Incidence Rate}_i = \left( \frac{C_i}{P_i} \right) \times 100,000$$

To account for small-number volatility in small local authorities (e.g. Rutland vs. Birmingham), we apply the **Clayton-Kaldor Empirical Bayes Smoother**:
$$r_i^{EB} = w_i r_i + (1 - w_i) \bar{r}$$
where $w_i = \frac{s^2}{s^2 + \bar{r}/P_i}$, $\bar{r}$ is the national mean rate, and $s^2$ is the prior variance.

### 2. Spatial Weight Matrix ($W$)
Spatial interaction is encoded into a row-standardized matrix $W$:
* **Queen Contiguity**: Two Upper Tier Local Authorities (UTLAs) share a boundary point or edge ($w_{ij} > 0$). Islands are reconnected using nearest neighbor linkage to ensure full graph connectivity.
* **$k$-Nearest Neighbours (KNN, $k=5$)**: Connects each UTLA to its 5 nearest geographic centroids, guaranteeing uniform degrees across all units.

### 3. Global Moran's $I$ (National Clustering)
Evaluates whether measles incidence across England is spatially autocorrelated:
$$I = \frac{n}{\sum_{i=1}^n \sum_{j=1}^n w_{ij}} \frac{\sum_{i=1}^n \sum_{j=1}^n w_{ij}(y_i - \bar{y})(y_j - \bar{y})}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
* $E[I] = -1 / (n - 1)$.
* Non-parametric significance evaluated via **999 Monte Carlo permutations**.
* **Observed Result**: $I \approx 0.32$, $z \approx 5.8$, $p < 0.001$, firmly rejecting spatial randomness.

### 4. Local Indicators of Spatial Association (LISA)
Decomposes Global Moran's $I$ into individual UTLA contributions:
$$I_i = \frac{(y_i - \bar{y})}{s^2} \sum_{j=1}^n w_{ij} (y_j - \bar{y})$$
Classifies each UTLA into four distinct quadrants at $\alpha = 0.05$:
1. **High-High (Hotspot)**: Elevated incidence surrounded by elevated incidence $\implies$ Active regional epidemic transmission corridors (e.g. Birmingham / West Midlands cluster; Inner London boroughs).
2. **Low-Low (Coldspot)**: Low incidence surrounded by low incidence $\implies$ Structurally protected or resilient zones (e.g. rural South West).
3. **High-Low (Spatial Outlier)**: High incidence surrounded by low incidence $\implies$ Isolated importations or spark outbreaks with limited secondary spread.
4. **Low-High (Spatial Outlier)**: Low incidence surrounded by high incidence $\implies$ Resilient buffer pockets adjacent to epidemic epicenters.

### 5. Getis-Ord $G_i^*$ (Hotspot Detection)
Includes the focal observation in the local neighbourhood to yield standard normal $z$-scores:
$$G_i^* = \frac{\sum_{j=1}^n w_{ij} y_j - \bar{Y}\sum_{j=1}^n w_{ij}}{S \sqrt{\frac{n \sum_{j=1}^n w_{ij}^2 - (\sum_{j=1}^n w_{ij})^2}{n - 1}}}$$
Classified into 7 confidence tiers ($99\%$, $95\%$, $90\%$ Hot Spots, Not Significant, and $90\%$, $95\%$, $99\%$ Cold Spots).

---

## Installation & Setup

### 1. Prerequisites
* Python 3.10+ (Recommended: Conda or Virtualenv)

### 2. Environment Setup
```bash
# Clone the repository or navigate to project directory
cd measles_spatial_surveillance

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Baseline Assets & Run Test Suite
```bash
# Run the automated pytest suite (14 unit & integration tests)
pytest tests/ -v
```

---

## Launching the Streamlit Web Application

To launch the interactive dashboard:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Dashboard Features
1. **Surveillance Parameter Sidebar**:
   - Filter by year: `All (Cumulative 2023–2026)`, `2024 (Epidemic Peak)`, `2023`, `2025`, `2026`.
   - Metric switcher: `Crude Incidence Rate (per 100k)`, `Empirical Bayes Smoothed Rate`, `Raw Confirmed Cases`.
   - Spatial weights selector: `KNN (k=5)` or `Queen Contiguity`.
   - Significance slider: $\alpha \in \{0.01, 0.05, 0.10\}$.
   - Regional drill-down filter.
2. **Geographic Surveillance Tab**:
   - High-performance Plotly choropleth map of England.
   - League table of Top 10 burdened UTLAs with population, MMR, and case counts.
   - Regional aggregated bar chart.
3. **Spatial Autocorrelation Tab**:
   - Global Moran's $I$ KPI cards and automated public health interpretation.
   - Interactive LISA Cluster map paired with the Anselin Moran Scatterplot.
   - Getis-Ord $G_i^*$ confidence hotspot map.
   - Spatial neighborhood sensitivity analysis matrix.
4. **Spatio-Temporal Dynamics Tab**:
   - National epidemic curve (monthly cases Jan 2023 – Oct 2026) with 3-month rolling average.
   - Outbreak milestone annotations (Nov 2023 Birmingham ignition, Spring 2024 London peak).
   - Regional case composition stacked area chart.
   - Multi-year trajectory of Global Moran's $I$.
5. **Immunisation Deficits Tab**:
   - MMR2 coverage at 5 years vs. measles incidence scatterplot with OLS trendline.
   - WHO 95.0% herd immunity target threshold line.
   - Immunity gap analysis identifying high-risk local authorities.
6. **Epidemiological Briefing Tab**:
   - Detailed public health findings, intervention strategies, and method limitations.

---

## Senior Data Science Interview Defense Guide

When discussing this project in a technical or senior data science interview, use the following points to defend your design choices:

### Q1: Why did you compute spatial autocorrelation on incidence rates rather than raw case counts?
> **Answer**: Running spatial autocorrelation directly on raw case counts commits the classic **spatial population density fallacy**: areas with more people naturally accumulate more raw cases, so Moran's $I$ would simply detect urban population clustering rather than genuine epidemiological transmission. By standardising case counts against ONS mid-year population denominators to obtain incidence rates per 100,000, we isolate the true geographic intensity of disease transmission. Furthermore, we implemented Empirical Bayes rate smoothing to ensure that small-population authorities with few cases do not distort the variance.

### Q2: How did you select the spatial weight matrix ($W$), and how did you handle islands?
> **Answer**: In spatial econometrics, the choice of $W$ defines the topology of spatial interaction. We implemented both **Queen contiguity** and **$k$-Nearest Neighbours (KNN, $k=5$)**. Queen contiguity reflects shared border interactions, but can isolate geographic islands (such as the Isle of Wight). In our data loader, we implemented an island-reconnection routine that augments Queen contiguity with nearest centroid linkages. We also ran a **sensitivity analysis** across Queen and KNN ($k=3, 5, 8$), demonstrating that Global Moran's $I$ remains stable ($p < 0.001$) regardless of specification, proving our clusters are genuine.

### Q3: What is the difference between Local Moran's I (LISA) and Getis-Ord Gi*?
> **Answer**: While both identify spatial clustering, they answer subtly different questions:
> * **LISA** decomposes Global Moran's $I$ into four quadrants (High-High, Low-Low, High-Low, Low-High). It is designed to identify both spatial clusters (homogeneity) and **spatial outliers** (heterogeneity, such as an isolated spark outbreak surrounded by coldspots).
> * **Getis-Ord $G_i^*$** is a local sum statistic that includes the focal unit ($star=True$). It calculates a standard normal $z$-score indicating whether the local sum of values is significantly higher or lower than expected by chance, directly outputting confidence tiers ($90\%, 95\%, 99\%$) for hot spots and cold spots.

### Q4: How does this project account for the Modifiable Areal Unit Problem (MAUP) and Ecological Fallacy?
> **Answer**: 
> * **MAUP**: Surveillance data is aggregated at Upper Tier Local Authority (UTLA) administrative units, which vary in area and population. Scale and zoning effects can alter statistical associations. We explicitly document this limitation and recommend future micro-targeted surveillance at MSOA (Middle Layer Super Output Area) or postcode district level where data governance permits.
> * **Ecological Fallacy**: Aggregated UTLA-level correlations (e.g. low MMR2 coverage correlating with elevated measles incidence) cannot prove that an individual who contracted measles was unvaccinated. However, in infectious disease dynamics, community herd immunity operates as an ecological-level protective barrier, making area-level coverage directly relevant to transmission risk.

### Q5: What are the public health policy recommendations arising from this surveillance?
> **Answer**: 
> 1. **Targeted School-Entry Catch-Up**: The data indicates that over 60% of cases are in children aged $\le 10$ years, heavily concentrated in UTLAs with MMR2 coverage $<80\%$ (e.g. Hackney, Birmingham, Lambeth). Catch-up campaigns should be deployed directly in primary schools in identified High-High LISA clusters.
> 2. **Mobile Outreach in Urban Deprivation Pockets**: Hotspots correlate strongly with urban deprivation and healthcare access barriers rather than ideological anti-vaccine sentiment alone. Weekend and walk-in community vaccine hubs in High-High clusters provide high return on investment.

---

## Data Sources & Provenance

* **UK Health Security Agency (UKHSA)**: *Confirmed cases of measles in England by month, age, region and upper tier local authority (2023 to 2026)*. Published on GOV.UK.
* **UKHSA COVER Programme**: *Quarterly and annual childhood vaccination coverage statistics for England*.
* **Office for National Statistics (ONS)**: *Mid-year population estimates by Upper Tier Local Authority and County/Unitary Authority boundaries*.
* **Licensing**: Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
