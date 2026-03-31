# Air Quality & Public Health Impact Pipeline — EEA + ERA5 + EUROSTAT
### *Predicting dangerous pollution events and estimating their causal effect on respiratory & cardiovascular mortality — France, Spain, Belgium, Netherlands & Germany*

---

> **English version below** | **Versión en español más abajo**

---

## English

> **Work in progress.** This project is under active development. Some pipeline stages, analyses, and documentation sections are incomplete or subject to change.

> **Engineering philosophy.** Beyond the analytical goals, this project is built with a deliberate focus on *good programming practices and MLOps principles* — aiming for a pipeline that is reliable, reusable, maintainable, flexible, and fully reproducible. This includes modular code structure, version-controlled experiments, schema validation, automated testing, containerisation, and CI/CD-driven deployment.

### Business Case & Problem Statement

Air pollution is the single largest environmental health risk in Europe, responsible for over <u>*400,000 premature deaths per year*</u> across the EU according to the European Environment Agency. Despite significant progress in reducing emissions over recent decades, pollutant concentrations still exceed both EU legal limits and the stricter WHO guidelines in many urban and industrial areas.

The economic cost is equally severe: lost working days, hospital admissions, and long-term chronic disease place hundreds of billions of euros of pressure on European healthcare systems annually, with the burden falling disproportionately on lower-income populations living near industrial zones, highways, and ports.

Despite the availability of dense monitoring networks, most public-facing tools offer only *current* air quality readings — they do not *predict* when conditions will become dangerous, nor do they *quantify* the downstream health burden those conditions cause. A 24 to 48-hour advance warning of a high-pollution event allows individuals, hospitals, and municipal governments to take preventive action: issuing health advisories, adjusting industrial operations, activating low-emission zones, or simply advising vulnerable populations to stay indoors.

This project addresses both gaps by building an *end-to-end machine learning and causal inference pipeline* that ingests historical atmospheric, pollution, and health data across Europe, engineers predictive features at scale, estimates the causal effect of pollution on mortality and hospitalizations, and serves real-time forecasts through a production-grade API.

---

### Geographic Scope & Design Rationale

The pipeline covers five countries: **France, Spain, Belgium, the Netherlands, and Germany**, over a **five-year period (2019–2023)**.

**Why these five countries?**

This selection is the result of a deliberate trade-off between analytical richness, computational feasibility, and causal identification power.

At one extreme, a single small country (e.g. Belgium alone) would be computationally trivial but analytically weak: too few NUTS3 regions for credible DiD control groups, limited geographic heterogeneity for the Causal Forest, and no transboundary pollution signal. At the other extreme, all of Europe would maximise statistical power but require cloud infrastructure (Databricks, EMR) and download volumes in the hundreds of gigabytes — unnecessary for a focused research pipeline.

The five-country scope hits the optimal point:

| Criterion | Single country | 5-country scope | All Europe |
|---|---|---|---|
| Estimated EEA stations | ~70 | ~1,000–1,200 | ~5,000+ |
| Estimated daily rows (5 years) | ~130K | ~2M | ~100M+ |
| Pandas sufficient? | ✅ | ✅ (borderline) | ❌ |
| Spark justified? | ❌ overkill | ✅ demonstrable | ✅ necessary |
| DiD control groups | ⚠️ few | ✅ rich | ✅ rich |
| Transboundary pollution signal | ❌ | ✅ | ✅ |

**Why Spark at ~2M rows?**

At 2 million rows, pandas is technically sufficient. Spark is retained for two explicit reasons: (1) it demonstrates production-grade data engineering skills, which is part of the project's purpose; and (2) the pipeline is designed to scale — adding more countries or years requires only a configuration change, not a rewrite. A note to this effect is more intellectually honest than pretending Spark is strictly necessary at this scale.

**Why these specific countries?**

France, Spain, Belgium, the Netherlands, and Germany form a geographically coherent bloc with documented transboundary pollution interactions — industrial emissions from the Rhine-Ruhr valley affect Belgium and the Netherlands; Saharan dust episodes affect Spain and France simultaneously. This cross-border structure is exactly what the DiD and IV analyses need to identify causal effects that go beyond what within-country variation alone could support.

---

### Why Combine EEA, ERA5, and Health Data?

These three data sources are individually powerful but tell incomplete stories on their own.

**EEA Air Quality e-Reporting** provides ground-level measurements of five key pollutants — PM2.5, PM10, ozone (O₃), nitrogen dioxide (NO₂), and sulphur dioxide (SO₂) — collected from thousands of official monitoring stations across all EU member states and several neighboring countries. It tells us *what the air contains*, but not *why* conditions changed, *what will happen next*, or *how many people will be harmed*.

**ERA5 (Copernicus/ECMWF)** is the European reference for atmospheric reanalysis, providing continuous gridded data — temperature, wind speed, humidity, atmospheric pressure, and boundary layer height — at 0.25° resolution from 1940 to present. It tells us about the *atmospheric conditions* that drive pollutant behavior: temperature inversions that trap particles near the ground, wind patterns that disperse or concentrate pollutants across national borders, and precipitation events that temporarily clean the air. As a 100% European dataset produced by the European Centre for Medium-Range Weather Forecasts.

**EUROSTAT and WHO European Mortality Database** provide the health outcome layer: weekly mortality by cause, age, and NUTS3 region, and hospital admissions for respiratory and cardiovascular conditions. They tell us *how many people were harmed* — the outcome that gives the entire pipeline its public health meaning.

Combining all three allows the pipeline to both predict dangerous pollution events and rigorously estimate their causal downstream effects on human health.

The transboundary nature of European air pollution — where industrial emissions from one country affect air quality in neighboring ones — makes the multi-country, multi-source approach especially relevant. This is also what justifies Apache Spark: while ~2M rows at this geographic scope could technically be handled by pandas, Spark is retained deliberately to demonstrate production-grade data engineering and to ensure the pipeline scales without rewriting if the scope expands (see Geographic Scope & Design Rationale above).

---

### Why PM2.5 as the Prediction Target?

The pipeline downloads and processes all five EEA pollutants — PM2.5, PM10, NO₂, O₃, and SO₂. In the modelling phase, **PM2.5 is chosen as the primary prediction target**, but the other four pollutants are retained as input features rather than discarded. This is a deliberate design choice grounded in both epidemiology and machine learning.

**PM2.5 as target — the public health case:**

PM2.5 (fine particulate matter with diameter ≤ 2.5 µm) is the pollutant with the strongest and most consistent evidence of harm across medical literature. Unlike PM10, which is largely filtered in the upper respiratory tract, PM2.5 particles penetrate deep into the lungs and cross into the bloodstream, causing cardiovascular disease, respiratory illness, and neurological damage. The WHO 2021 Air Quality Guidelines revised the PM2.5 annual mean guideline down to 5 µg/m³ — the most stringent update in two decades — precisely because the evidence base for harm at low concentrations is now overwhelming.

For a predictive warning system, PM2.5 exceedance events are the most actionable signal: they trigger health advisories, activate emergency protocols in hospitals, and determine legal compliance for industrial operators.

**PM2.5 as target — the modelling case:**

PM2.5 has the best statistical properties for learning: strong temporal autocorrelation (yesterday's PM2.5 is highly predictive of tomorrow's), clear seasonal patterns driven by heating seasons and temperature inversions, and a well-documented physical relationship with meteorological variables. This makes lag features and rolling windows particularly effective, and SHAP values particularly interpretable.

**The other pollutants as features — not wasted:**

PM10, NO₂, O₃, and SO₂ are retained as input features with the following physical justifications:

| Feature | Role | Physical rationale |
|---|---|---|
| `pm10_value` | Same-day feature + t-1 lag | PM10 is a superset of PM2.5 — strong collinear predictor. High PM10 days almost always imply elevated PM2.5. |
| `no2_value` | Same-day feature + t-1 lag | NO₂ is a proxy for traffic and combustion intensity — the same processes that emit PM2.5. |
| `o3_value` | Same-day feature + t-1 lag | O₃ tends to anticorrelate with PM2.5. Low O₃ in summer can signal stagnant air that accumulates particles. |
| `so2_value` | Same-day feature + t-1 lag | SO₂ indicates industrial activity that co-emits fine particles. Persistent SO₂ episodes often precede PM2.5 exceedance events. |

Same-day pollutant values are not leaky: at prediction time (end of day *t*), all five pollutant readings for day *t* are already known from the monitoring network.

---

### Causal Inference Layer — Why It Matters

Most air quality pipelines stop at prediction: *will PM2.5 be dangerous tomorrow?* This project goes further by asking the causal question: *how many additional deaths and hospitalizations does a PM2.5 exceedance event actually cause?*

This distinction matters operationally. A predictive model tells hospitals to prepare; a causal estimate tells policymakers how many lives a stricter emission standard would save. Both questions require the same data infrastructure — but different analytical tools.

Three causal analyses are embedded in the pipeline:

**Analysis 1 — Difference-in-Differences: PM2.5 episodes and respiratory mortality**

Using NUTS3 regions that exceed the PM2.5 threshold as the treated group and comparable regions that do not as controls, this analysis estimates the average increase in weekly respiratory deaths in the days following a pollution episode, controlling for seasonality and long-run trends through ERA5 meteorological covariates.

**Analysis 2 — Instrumental Variables: PM2.5 and cardiovascular hospitalizations**

Boundary layer height from ERA5 serves as the instrument: temperature inversions trap pollutants near the ground (strongly affecting PM2.5 levels) without directly affecting cardiovascular health through any channel other than pollution. This allows identification of the causal effect of PM2.5 on hospital admissions even in the presence of unmeasured confounders.

**Analysis 3 — Heterogeneous Treatment Effects: who suffers most? (EconML / Causal Forest)**

Causal Forest estimates conditional average treatment effects (CATE) by NUTS3 region, season, and age group, revealing whether the health impact of PM2.5 is larger in Eastern Europe, in winter, or among elderly populations — questions that a single average effect cannot answer.

---

### Expected Outcomes & Outputs

| Output | Description |
|---|---|
| **Processed feature store** | Parquet dataset partitioned by country, year, and pollutant — clean, spatially joined with ERA5 grid, and ready for modeling |
| **Aggregation tables** | City-level summaries: days above EU/WHO thresholds, seasonal patterns, cross-border pollution correlations |
| **Trained ML models** | LightGBM and XGBoost classifiers predicting PM2.5 threshold exceedance at 24h (t+1) and 48h (t+2) horizons |
| **SHAP analysis** | Interpretability report identifying which features most strongly drive pollution events by region and season |
| **Causal estimates** | DiD coefficients, IV estimates, and CATE maps quantifying the health burden of PM2.5 exceedance events |
| **Production API** | FastAPI endpoints serving predictions and causal health impact estimates |
| **Orchestrated pipeline** | Prefect DAG automating the full flow from ingestion to model retraining, scheduled weekly |
| **Interactive dashboard** | Plotly Dash application showing Europe-wide PM2.5 trends, mortality burden maps, and model performance metrics |

---

### Modelling Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Prediction target | PM2.5 exceedance (binary) | Highest health impact; strong autocorrelation; actionable threshold |
| Prediction horizons | t+1 (24h) and t+2 (48h) | Operationally meaningful for health advisories and industrial planning |
| Co-pollutants | Features, not targets | Retain NO₂, O₃, PM10, SO₂ as inputs — they encode emission and atmospheric state |
| Meteorological source | ERA5 (Copernicus/ECMWF) | 100% European; continuous grid; superior spatial coverage over station networks |
| Priority metric | Recall (≥ 90%) | Missing a danger event (false negative) is worse than a false alarm |
| Class imbalance | `scale_pos_weight` in LightGBM/XGBoost | Adjusts for the fact that danger days are a minority of observations |
| Validation strategy | Strict temporal split | Random splits cause data leakage via lag features |
| Decision threshold | Tuned per model | Set to achieve ≥ 90% recall on validation set, not fixed at 0.5 |
| Causal instrument | ERA5 boundary layer height | Affects PM2.5 via thermal inversions; no direct health pathway |
| Causal heterogeneity | Causal Forest (EconML) | Estimates CATE by region, season, and age group |

---

### Methodology

**Phase 1 — Ingestion & Exploratory Analysis**

Raw data is downloaded programmatically from the EEA Air Quality Download Service, the Copernicus Climate Data Store (ERA5 via `cdsapi`), EUROSTAT (via the `eurostat` Python package), and the WHO European Mortality Database. EDA documents data coverage, missing value patterns, distribution shapes, class imbalance, seasonal structure, cross-pollutant correlations, and autocorrelation of PM2.5 across countries. Health data EDA covers mortality trends by cause, age, country, and season.

**Phase 2 — Large-Scale Processing with Apache Spark**

PySpark jobs handle cleaning, spatial joining (ERA5 grid interpolation to EEA station coordinates, replacing the Haversine nearest-neighbor approach used with NOAA point stations), and feature engineering: PM2.5 and co-pollutant lags, rolling windows, calendar features, derived meteorological indices, and health outcome joins at weekly NUTS3 granularity. Outputs are partitioned Parquet files.

**Phase 3 — Pipeline Orchestration with Prefect**

All steps are wrapped in a Prefect flow with dependency management, schema validation gates, retry logic with exponential backoff, and weekly scheduling. The pipeline runs incrementally — only new data is downloaded on each run. A flag file triggers model retraining only when new data arrived.

**Phase 4 — Predictive Modeling**

Binary classification (PM2.5 > 25 µg/m³ at t+1 and t+2) using LightGBM (primary), XGBoost (comparative), and Logistic Regression (interpretable baseline). Decision threshold tuned to achieve ≥ 90% recall. SHAP values provide full feature interpretability. Country-level performance breakdown identifies regional weaknesses.

**Phase 5 — Causal Inference**

Three causal analyses using `linearmodels` (DiD, IV) and `econml` (Causal Forest). Analyses are implemented as standalone notebooks with full narrative documentation of assumptions, identification strategy, and robustness checks. Results feed a `/health-impact` API endpoint that returns causal mortality and hospitalization estimates alongside the predictive alert.

**Phase 6 — Production Deploy**

FastAPI application containerised with Docker (multi-stage build, non-root user). CI/CD via GitHub Actions: tests on every PR, automatic deploy to Render.com on push to main. Endpoints: `/predict`, `/predict/batch`, `/health-impact`, `/health`, `/metrics`.

---

### Tech Stack

| Layer | Tool |
|---|---|
| Ingestion — air quality | Python, `airbase` |
| Ingestion — meteorology | Python, `cdsapi` (ERA5/Copernicus) |
| Ingestion — health | Python, `eurostat` package, `requests` (WHO) |
| Processing | Apache Spark (PySpark), `xarray` (NetCDF → Parquet) |
| Orchestration | Prefect |
| Experiment tracking | MLflow — logs hyperparameters, metrics, and model artefacts for every training run |
| Modeling | LightGBM, XGBoost, scikit-learn, SHAP |
| Causal inference | EconML, DoWhy, `linearmodels` |
| Visualization | Plotly Dash |
| API | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deploy | Render.com (free tier) |

---

### Visualisation

**Development — Interactive Dashboard (`dashboard/app.py`)**

A Plotly Dash application intended for exploratory analysis, internal communication of results, and demos. It is not part of the production deployment. Three panels:

- *Europe map*: PM2.5 exceedance days by NUTS3 region with a time slider, showing where and when dangerous pollution events concentrate across the continent.
- *Prediction panel*: observed vs predicted PM2.5 for a selected monitoring station, t+1/t+2 alert probability, and a SHAP bar chart explaining which features are driving the current prediction.
- *Causal panel*: CATE choropleth map of health impact by region, DiD event-study plots (treated vs control regions around pollution episodes), and IV estimates with confidence intervals by country.

> A `06_results_visualization.ipynb` notebook covering static model outputs (ROC curves, calibration plots, SHAP summary plots) is planned for reproducible offline review.

**Production — REST API (`api/main.py`)**

The only production-facing surface. A FastAPI application containerised with Docker, deployed via CI/CD to Render.com, and designed for integration with external systems (hospitals, municipal governments, monitoring platforms). Endpoints:

| Endpoint | Description |
|---|---|
| `POST /predict` | PM2.5 exceedance probability at t+1 and t+2 for a single station |
| `POST /predict/batch` | Same for multiple stations in one call |
| `GET /health-impact` | Causal estimate of additional deaths and hospitalisations given a pollution episode |
| `GET /metrics` | Live model performance metrics (recall, precision, prediction volume) |
| `GET /health` | Service health check |

---

### Project Structure

```
eea-era5-air-quality-europe/
├── data/                               # excluded from git (.gitignore)
│   ├── raw/
│   │   ├── eea/                        # verified + unverified Parquet
│   │   ├── era5/                       # NetCDF by month and year
│   │   ├── eurostat/
│   │   │   ├── demo_r_mweek3/          # weekly mortality by NUTS3
│   │   │   ├── hlth_cd_aro/            # deaths by respiratory/cardiovascular cause
│   │   │   └── hlth_co_hospit/         # hospital admissions by cause
│   │   └── who/                        # WHO European Mortality Database
│   └── processed/
│       ├── eea/                        # cleaned EEA, partitioned
│       ├── era5/                       # cleaned ERA5, regridded to EEA stations
│       ├── health/                     # cleaned mortality + hospitalization data
│       ├── features/                   # feature store (final ML + causal input)
│       └── aggregations/               # city-level yearly summaries + health KPIs
├── notebooks/
│   ├── 01_eda.ipynb                    # EDA: pollution, meteorology, health outcomes
│   ├── 02_feature_validation.ipynb     # feature store QA + quick model experiment
│   ├── 03_causal_did.ipynb             # DiD: PM2.5 episodes → respiratory mortality
│   ├── 04_causal_iv.ipynb              # IV: PM2.5 → cardiovascular hospitalizations
│   └── 05_causal_heterogeneity.ipynb   # Causal Forest: CATE by region, age, season
├── src/
│   ├── utils/                              # shared helpers — imported by every module
│   │   ├── paths.py                        # single source of truth for all filesystem paths — change here, change everywhere
│   │   ├── logging_config.py               # logger configured once, imported across all modules — avoids duplicate handlers
│   │   └── retry.py                        # reusable exponential-backoff decorator — EEA and ERA5 endpoints fail transiently
│   ├── ingestion/
│   │   ├── download_eea.py                 # downloads PM2.5/PM10/NO2/O3/SO2 Parquet files from EEA bulk service
│   │   ├── download_era5.py                # requests monthly NetCDF files from Copernicus CDS (temperature, wind, BLH…)
│   │   ├── download_eurostat.py            # fetches three health outcome tables via the eurostat Python package
│   │   ├── download_who.py                 # downloads WHO European Mortality Database ZIP and extracts CSVs
│   │   └── validate_downloads.py           # schema + size checks; writes flag files that gate the Spark jobs
│   ├── spark/
│   │   ├── spark_clean_eea.py              # cleans and repartitions EEA Parquet by country and year
│   │   ├── spark_clean_era5.py             # converts NetCDF → Parquet and interpolates ERA5 grid to EEA station coords
│   │   ├── spark_clean_health.py           # harmonises EUROSTAT and WHO mortality/hospitalisation tables
│   │   └── spark_join_features.py          # joins all sources into the final feature store (ML + causal input)
│   ├── causal/
│   │   ├── did_analysis.py                 # Difference-in-Differences: PM2.5 episodes → respiratory mortality
│   │   ├── iv_analysis.py                  # Instrumental Variables: boundary layer height as instrument for PM2.5
│   │   └── heterogeneous_effects.py        # Causal Forest (EconML): CATE by region, season, and age group
│   ├── model/
│   │   ├── train.py                        # trains LightGBM and XGBoost classifiers; logs runs to MLflow
│   │   └── evaluate.py                     # threshold tuning, recall/precision curves, SHAP values, country breakdown
│   └── pipeline/
│       └── flow.py                         # Prefect DAG — wires all tasks, manages flags, schedules weekly retraining
├── api/
│   └── main.py                             # FastAPI app — /predict, /predict/batch, /health-impact, /metrics endpoints
├── dashboard/
│   └── app.py                              # Plotly Dash — Europe-wide PM2.5 trends, mortality maps, model performance
├── tests/
│   └── test_api.py                         # API integration tests run on every PR via GitHub Actions
├── models/                             # saved model artifacts (excluded from git)
├── Dockerfile                          # API production image
├── Dockerfile.pipeline                 # Pipeline image (Spark + Prefect)
├── docker-compose.yml
├── prefect.yaml
├── requirements-ingestion.txt
├── requirements-spark.txt
├── requirements-model.txt
├── requirements-causal.txt
├── requirements-api.txt
├── requirements-dashboard.txt
└── requirements-pipeline.txt
```

---

### Quick Start

```bash
# 1. Install ingestion dependencies
pip install -r requirements-ingestion.txt

# 2. Configure ERA5 access (one-time setup)
# Register at https://cds.climate.copernicus.eu and save your key to ~/.cdsapirc

# 3. Download data (dry-run first to check sizes)
python src/ingestion/download_eea.py --dry-run
python src/ingestion/download_era5.py --dry-run
python src/ingestion/download_eurostat.py --dry-run

# 4. Download for real
python src/ingestion/download_eea.py
python src/ingestion/download_era5.py
python src/ingestion/download_eurostat.py
python src/ingestion/download_who.py

# 5. Validate downloads
python src/ingestion/validate_downloads.py

# 6. Run Spark jobs
pip install -r requirements-spark.txt
spark-submit src/spark/spark_clean_eea.py
spark-submit src/spark/spark_clean_era5.py
spark-submit src/spark/spark_clean_health.py
spark-submit src/spark/spark_join_features.py

# 7. Train models
pip install -r requirements-model.txt
python src/model/train.py
python src/model/evaluate.py

# 8. Run causal analyses (see notebooks/ for narrative documentation)
pip install -r requirements-causal.txt
jupyter notebook notebooks/03_causal_did.ipynb

# 9. Run API locally
pip install -r requirements-api.txt
uvicorn api.main:app --reload
# Swagger UI → http://localhost:8000/docs

# 10. Run dashboard
pip install -r requirements-dashboard.txt
python dashboard/app.py
# Dashboard → http://localhost:8050
```

---

### Data Sources

| Source | Dataset | URL |
|---|---|---|
|  EEA | Air Quality e-Reporting | https://www.eea.europa.eu/en/datahub/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d |
|  Copernicus/ECMWF | ERA5 Reanalysis | https://cds.climate.copernicus.eu |
|  EUROSTAT | Weekly mortality by NUTS3 (`demo_r_mweek3`) | https://ec.europa.eu/eurostat |
|  EUROSTAT | Deaths by respiratory/cardiovascular cause (`hlth_cd_aro`) | https://ec.europa.eu/eurostat |
|  EUROSTAT | Hospital admissions (`hlth_co_hospit`) | https://ec.europa.eu/eurostat |
|  WHO Europa | European Mortality Database | https://gateway.euro.who.int/en/datasets/european-mortality-database |

---
---

## Español

> **Proyecto en curso.** Este proyecto está en desarrollo activo. Algunas fases del pipeline, análisis y secciones de documentación están incompletas o sujetas a cambios.

> **Filosofía de ingeniería.** Más allá de los objetivos analíticos, este proyecto se construye con un enfoque deliberado en *buenas prácticas de programación y principios MLOps* — buscando un pipeline que sea fiable, reutilizable, mantenible, flexible y completamente reproducible. Esto incluye estructura de código modular, experimentos con control de versiones, validación de esquemas, tests automatizados, contenerización y despliegue mediante CI/CD.

### Caso de Negocio y Planteamiento del Problema

La contaminación del aire es el mayor riesgo ambiental para la salud en Europa, responsable de más de <u>*400.000 muertes prematuras al año*</u> en la UE según la Agencia Europea de Medio Ambiente. A pesar del progreso significativo en la reducción de emisiones, las concentraciones de contaminantes siguen superando tanto los límites legales de la UE como las directrices más estrictas de la OMS en muchas zonas urbanas e industriales — especialmente en Europa Central y del Este, el Valle del Po en Italia y las ciudades del oeste con alto tráfico.

La mayoría de las herramientas públicas solo ofrecen lecturas *actuales* de calidad del aire — no *predicen* cuándo las condiciones se volverán peligrosas, ni *cuantifican* la carga sanitaria que esas condiciones generan. Un aviso de 24 a 48 horas antes de un evento de alta contaminación permite a personas, hospitales y gobiernos municipales tomar medidas preventivas: emitir alertas sanitarias, ajustar operaciones industriales, activar zonas de bajas emisiones, o simplemente indicar a poblaciones vulnerables que permanezcan en interiores.

Este proyecto aborda ambas brechas construyendo un *pipeline end-to-end de machine learning e inferencia causal* que ingesta datos históricos atmosféricos, de contaminación y de salud a escala europea, genera features predictivas a gran escala, estima el efecto causal de la contaminación sobre la mortalidad y hospitalizaciones, y sirve predicciones en tiempo real a través de una API lista para producción.

---

### Ámbito Geográfico y Justificación del Diseño

El pipeline cubre cinco países: **Francia, España, Bélgica, Países Bajos y Alemania**, durante un **periodo de cinco años (2019–2023)**.

**¿Por qué estos cinco países?**

Esta selección es el resultado de un compromiso deliberado entre riqueza analítica, viabilidad computacional y potencia de identificación causal.

En un extremo, un único país pequeño (por ejemplo, solo Bélgica) sería trivial computacionalmente pero débil analíticamente: demasiado pocas regiones NUTS3 para grupos de control DiD creíbles, heterogeneidad geográfica limitada para el Causal Forest, y sin señal de contaminación transfronteriza. En el otro extremo, toda Europa maximizaría la potencia estadística pero requeriría infraestructura cloud (Databricks, EMR) y volúmenes de descarga de cientos de gigabytes — innecesario para un pipeline de investigación enfocado.

El ámbito de cinco países alcanza el punto óptimo:

| Criterio | Un país | 5 países | Europa completa |
|---|---|---|---|
| Estaciones EEA estimadas | ~70 | ~1.000–1.200 | ~5.000+ |
| Filas diarias estimadas (5 años) | ~130K | ~2M | ~100M+ |
| ¿Pandas suficiente? | ✅ | ✅ (al límite) | ❌ |
| ¿Spark justificado? | ❌ excesivo | ✅ demostrable | ✅ necesario |
| Grupos de control DiD | ⚠️ pocos | ✅ ricos | ✅ ricos |
| Señal de contaminación transfronteriza | ❌ | ✅ | ✅ |

**¿Por qué Spark con ~2M filas?**

Con 2 millones de filas, pandas es técnicamente suficiente. Spark se mantiene por dos razones explícitas: (1) demuestra capacidades de ingeniería de datos a escala de producción, que forma parte del propósito del proyecto; y (2) el pipeline está diseñado para escalar — añadir más países o años requiere solo un cambio de configuración, no una reescritura. Reconocer esto explícitamente es más honesto intelectualmente que pretender que Spark es estrictamente necesario a esta escala.

**¿Por qué estos países concretos?**

Francia, España, Bélgica, Países Bajos y Alemania forman un bloque geográficamente coherente con interacciones de contaminación transfronteriza documentadas — las emisiones industriales del valle del Rin-Ruhr afectan a Bélgica y Países Bajos; los episodios de polvo del Sáhara afectan simultáneamente a España y Francia. Esta estructura transfronteriza es exactamente lo que los análisis DiD e IV necesitan para identificar efectos causales que vayan más allá de lo que la variación dentro de un solo país podría sustentar.

---

### ¿Por Qué Combinar EEA, ERA5 y Datos de Salud?

**EEA Air Quality e-Reporting** nos dice *qué contiene el aire* — niveles de PM2.5, PM10, O₃, NO₂ y SO₂ — pero no *por qué cambiaron las condiciones*, *qué pasará después*, ni *cuántas personas serán afectadas*.

**ERA5 (Copernicus/ECMWF)** es la referencia europea para el reanálisis atmosférico, proporcionando datos en grid continuo — temperatura, velocidad del viento, humedad, presión y altura de la capa límite — a resolución de 0,25° desde 1940. Nos habla de las *condiciones atmosféricas* que determinan el comportamiento de los contaminantes: inversiones térmicas que atrapan partículas cerca del suelo, patrones de viento que dispersan o concentran contaminantes a través de fronteras, y precipitaciones que limpian temporalmente el aire. Como dataset 100% europeo producido por el Centro Europeo de Predicción Meteorológica a Plazo Medio.

**EUROSTAT y la Base de Datos Europea de Mortalidad de la OMS** aportan la capa de resultados sanitarios: mortalidad semanal por causa, edad y región NUTS3, y hospitalizaciones por enfermedades respiratorias y cardiovasculares. Nos dicen *cuántas personas fueron dañadas* — el resultado que otorga al pipeline todo su significado de salud pública.

La naturaleza transfronteriza de la contaminación del aire europea — donde las emisiones industriales de un país afectan la calidad del aire de los países vecinos — hace que el enfoque multipaís y multifuente sea especialmente relevante. Esto también justifica el uso de Apache Spark: aunque ~2M filas en este ámbito geográfico podrían manejarse técnicamente con pandas, Spark se mantiene deliberadamente para demostrar ingeniería de datos a escala de producción y garantizar que el pipeline escale sin reescritura si el ámbito se amplía (ver Ámbito Geográfico y Justificación del Diseño).

---

### ¿Por Qué PM2.5 como Variable Objetivo?

**PM2.5 como target — el caso de salud pública:**

PM2.5 (partículas finas de diámetro ≤ 2,5 µm) es el contaminante con mayor evidencia científica de daño. A diferencia del PM10, que queda filtrado en el tracto respiratorio superior, las partículas finas penetran en los pulmones y cruzan al torrente sanguíneo, causando enfermedades cardiovasculares, respiratorias y neurológicas. La OMS revisó en 2021 su directriz anual de PM2.5 a 5 µg/m³ — la actualización más estricta en dos décadas.

**Los otros contaminantes como features — no desperdiciados:**

PM10, NO₂, O₃ y SO₂ se retienen como variables de entrada:

| Feature | Rol | Justificación física |
|---|---|---|
| `pm10_value` | Feature del día + lag t-1 | PM10 contiene PM2.5 como subconjunto — predictor colineal fuerte |
| `no2_value` | Feature del día + lag t-1 | Proxy de intensidad de tráfico y combustión |
| `o3_value` | Feature del día + lag t-1 | Anticorrelaciona con PM2.5; O₃ bajo en verano señala aire estancado |
| `so2_value` | Feature del día + lag t-1 | Indica actividad industrial que co-emite partículas finas |

---

### Capa de Inferencia Causal — Por Qué Importa

La mayoría de los pipelines de calidad del aire se detienen en la predicción: *¿será peligroso el PM2.5 mañana?* Este proyecto va más allá preguntando: *¿cuántas muertes y hospitalizaciones adicionales causa realmente un episodio de PM2.5?*

Esta distinción importa operativamente. Un modelo predictivo dice a los hospitales que se preparen; una estimación causal dice a los legisladores cuántas vidas salvaría una norma de emisión más estricta.

**Análisis 1 — DiD: episodios de PM2.5 y mortalidad respiratoria**
Regiones que superan el umbral (tratadas) vs regiones comparables que no lo superan (control), estimando el incremento en muertes respiratorias semanales en los días posteriores a un episodio.

**Análisis 2 — Variables Instrumentales: PM2.5 y hospitalizaciones cardiovasculares**
La altura de la capa límite de ERA5 actúa como instrumento: las inversiones térmicas afectan fuertemente los niveles de PM2.5 sin impactar directamente la salud cardiovascular por ninguna otra vía.

**Análisis 3 — Efectos Heterogéneos: ¿quién sufre más? (EconML / Causal Forest)**
Causal Forest estima efectos causales condicionales (CATE) por región NUTS3, estación y grupo de edad, revelando si el impacto sanitario del PM2.5 es mayor en Europa del Este, en invierno, o en mayores de 65 años.

---

### Decisiones de Diseño del Modelado

| Decisión | Elección | Justificación |
|---|---|---|
| Variable objetivo | Excedencia de PM2.5 (binario) | Mayor impacto sanitario; autocorrelación fuerte; umbral accionable |
| Horizontes | t+1 (24h) y t+2 (48h) | Útiles para avisos sanitarios y planificación industrial |
| Fuente meteorológica | ERA5 (Copernicus/ECMWF) | 100% europeo; grid continuo; cobertura superior a redes de estaciones |
| Co-contaminantes | Features, no targets | NO₂, O₃, PM10, SO₂ codifican estado de emisiones y atmósfera |
| Métrica prioritaria | Recall (≥ 90%) | Perder un evento peligroso es peor que una falsa alarma |
| Desbalanceo de clases | `scale_pos_weight` | Ajusta por la minoría de días peligrosos |
| Estrategia de validación | Corte temporal estricto | Los splits aleatorios causan data leakage vía lag features |
| Instrumento causal | Altura capa límite ERA5 | Afecta PM2.5 vía inversiones térmicas; sin vía directa a salud |
| Heterogeneidad causal | Causal Forest (EconML) | Estima CATE por región, estación y grupo de edad |

---

### Stack Tecnológico

| Capa | Herramienta |
|---|---|
| Ingesta — calidad del aire | Python, `airbase` |
| Ingesta — meteorología | Python, `cdsapi` (ERA5/Copernicus) |
| Ingesta — salud | Python, `eurostat` package, `requests` (WHO) |
| Procesamiento | Apache Spark (PySpark), `xarray` (NetCDF → Parquet) |
| Orquestación | Prefect |
| Seguimiento de experimentos | MLflow — registra hiperparámetros, métricas y artefactos del modelo en cada ejecución de entrenamiento |
| Modelado | LightGBM, XGBoost, scikit-learn, SHAP |
| Inferencia causal | EconML, DoWhy, `linearmodels` |
| Visualización | Plotly Dash |
| API | FastAPI |
| Containerización | Docker |
| CI/CD | GitHub Actions |
| Deploy | Render.com (free tier) |

---

### Visualización

**Desarrollo — Dashboard Interactivo (`dashboard/app.py`)**

Aplicación Plotly Dash destinada a exploración analítica, comunicación interna de resultados y demos. No forma parte del despliegue en producción. Tres paneles:

- *Mapa de Europa*: días de excedencia de PM2.5 por región NUTS3 con slider temporal, mostrando dónde y cuándo se concentran los episodios de contaminación peligrosa en el continente.
- *Panel predictivo*: PM2.5 real vs predicho para una estación seleccionada, probabilidad de alerta a t+1/t+2, y gráfico SHAP de barras que explica qué features están impulsando la predicción actual.
- *Panel causal*: mapa coropleta de CATE del impacto sanitario por región, gráficos de estudio de eventos DiD (regiones tratadas vs control alrededor de episodios de contaminación), y estimaciones IV con intervalos de confianza por país.

> Un notebook `06_results_visualization.ipynb` con outputs estáticos del modelo (curvas ROC, gráficos de calibración, SHAP summary plots) está planificado para revisión offline reproducible.

**Producción — API REST (`api/main.py`)**

La única superficie orientada a producción. Aplicación FastAPI contenerizada con Docker, desplegada mediante CI/CD en Render.com, y diseñada para integración con sistemas externos (hospitales, gobiernos municipales, plataformas de monitoreo). Endpoints:

| Endpoint | Descripción |
|---|---|
| `POST /predict` | Probabilidad de excedencia de PM2.5 a t+1 y t+2 para una estación |
| `POST /predict/batch` | Lo mismo para múltiples estaciones en una sola llamada |
| `GET /health-impact` | Estimación causal de muertes y hospitalizaciones adicionales dado un episodio de contaminación |
| `GET /metrics` | Métricas de rendimiento del modelo en tiempo real (recall, precisión, volumen de predicciones) |
| `GET /health` | Health check del servicio |

---

### Fuentes de Datos

| Fuente | Dataset | URL |
|---|---|---|
|  EEA | Air Quality e-Reporting | https://www.eea.europa.eu/en/datahub/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d |
|  Copernicus/ECMWF | ERA5 Reanalysis | https://cds.climate.copernicus.eu |
|  EUROSTAT | Mortalidad semanal NUTS3 (`demo_r_mweek3`) | https://ec.europa.eu/eurostat |
|  EUROSTAT | Muertes por causa respiratoria/cardiovascular (`hlth_cd_aro`) | https://ec.europa.eu/eurostat |
|  EUROSTAT | Hospitalizaciones (`hlth_co_hospit`) | https://ec.europa.eu/eurostat |
|  WHO Europa | European Mortality Database | https://gateway.euro.who.int/en/datasets/european-mortality-database |

---

### Metodología

**Fase 1 — Ingesta y Análisis Exploratorio**

Los datos se descargan de forma programática desde el servicio de descarga de calidad del aire de la EEA, el Copernicus Climate Data Store (ERA5 vía `cdsapi`), EUROSTAT (vía el paquete Python `eurostat`) y la Base de Datos Europea de Mortalidad de la OMS. El EDA documenta cobertura de datos, patrones de valores ausentes, distribuciones, desbalanceo de clases, estructura estacional, correlaciones entre contaminantes y autocorrelación del PM2.5 por país. El EDA de salud cubre tendencias de mortalidad por causa, edad, país y estación.

**Fase 2 — Procesamiento a Gran Escala con Apache Spark**

Los jobs de PySpark gestionan la limpieza, la unión espacial (interpolación del grid ERA5 a las coordenadas de las estaciones EEA) y la ingeniería de features: lags de PM2.5 y co-contaminantes, ventanas móviles, variables de calendario, índices meteorológicos derivados, y uniones con datos de salud a granularidad semanal NUTS3. Las salidas son archivos Parquet particionados.

**Fase 3 — Orquestación del Pipeline con Prefect**

Todos los pasos están encapsulados en un flow de Prefect con gestión de dependencias, validaciones de esquema, lógica de reintentos con backoff exponencial y programación semanal. El pipeline funciona de forma incremental — solo se descargan los datos nuevos en cada ejecución. Un fichero flag activa el reentrenamiento del modelo únicamente cuando llegaron datos nuevos.

**Fase 4 — Modelado Predictivo**

Clasificación binaria (PM2.5 > 25 µg/m³ en t+1 y t+2) usando LightGBM (modelo principal), XGBoost (comparativo) y Regresión Logística (baseline interpretable). El umbral de decisión se ajusta para alcanzar un recall ≥ 90%. Los valores SHAP proporcionan interpretabilidad completa de las features. Un desglose del rendimiento por país identifica debilidades regionales.

**Fase 5 — Inferencia Causal**

Tres análisis causales usando `linearmodels` (DiD, IV) y `econml` (Causal Forest). Los análisis se implementan como notebooks independientes con documentación narrativa completa de supuestos, estrategia de identificación y verificaciones de robustez. Los resultados alimentan un endpoint `/health-impact` de la API que devuelve estimaciones causales de mortalidad y hospitalización junto con la alerta predictiva.

**Fase 6 — Despliegue en Producción**

Aplicación FastAPI contenerizada con Docker (build multi-stage, usuario no-root). CI/CD mediante GitHub Actions: tests en cada PR, despliegue automático a Render.com en cada push a main. Endpoints: `/predict`, `/predict/batch`, `/health-impact`, `/health`, `/metrics`.

---

### Estructura del Proyecto

```
eea-era5-air-quality-europe/
├── data/                               # excluido de git (.gitignore)
│   ├── raw/
│   │   ├── eea/                        # Parquet verificado y no verificado
│   │   ├── era5/                       # NetCDF por mes y año
│   │   ├── eurostat/
│   │   │   ├── demo_r_mweek3/          # mortalidad semanal por NUTS3
│   │   │   ├── hlth_cd_aro/            # muertes por causa respiratoria/cardiovascular
│   │   │   └── hlth_co_hospit/         # hospitalizaciones por causa
│   │   └── who/                        # Base de Datos Europea de Mortalidad OMS
│   └── processed/
│       ├── eea/                        # EEA limpio y particionado
│       ├── era5/                       # ERA5 limpio, reproyectado a estaciones EEA
│       ├── health/                     # datos de mortalidad y hospitalización limpios
│       ├── features/                   # feature store (input final para ML y causal)
│       └── aggregations/               # resúmenes anuales por ciudad + KPIs de salud
├── notebooks/
│   ├── 01_eda.ipynb                    # EDA: contaminación, meteorología, salud
│   ├── 02_feature_validation.ipynb     # QA del feature store + experimento rápido
│   ├── 03_causal_did.ipynb             # DiD: episodios PM2.5 → mortalidad respiratoria
│   ├── 04_causal_iv.ipynb              # IV: PM2.5 → hospitalizaciones cardiovasculares
│   └── 05_causal_heterogeneity.ipynb   # Causal Forest: CATE por región, edad, estación
├── src/
│   ├── utils/                              # utilidades compartidas — importadas por todos los módulos
│   │   ├── paths.py                        # fuente única de verdad para todas las rutas — se cambia aquí, se cambia en todo
│   │   ├── logging_config.py               # logger configurado una vez, importado en todos los módulos — evita handlers duplicados
│   │   └── retry.py                        # decorador de reintentos con backoff exponencial — los endpoints de EEA y ERA5 fallan de forma transitoria
│   ├── ingestion/
│   │   ├── download_eea.py                 # descarga archivos Parquet de PM2.5/PM10/NO2/O3/SO2 desde el servicio bulk de la EEA
│   │   ├── download_era5.py                # solicita archivos NetCDF mensuales al CDS de Copernicus (temperatura, viento, BLH…)
│   │   ├── download_eurostat.py            # obtiene tres tablas de resultados sanitarios via el paquete Python eurostat
│   │   ├── download_who.py                 # descarga el ZIP de la Base de Datos de Mortalidad de la OMS y extrae los CSV
│   │   └── validate_downloads.py           # comprobaciones de esquema y tamaño; escribe flag files que bloquean los jobs de Spark
│   ├── spark/
│   │   ├── spark_clean_eea.py              # limpia y reparticiona los Parquet de EEA por país y año
│   │   ├── spark_clean_era5.py             # convierte NetCDF → Parquet e interpola el grid ERA5 a coordenadas de estaciones EEA
│   │   ├── spark_clean_health.py           # armoniza las tablas de mortalidad/hospitalización de EUROSTAT y OMS
│   │   └── spark_join_features.py          # une todas las fuentes en el feature store final (input para ML y causal)
│   ├── causal/
│   │   ├── did_analysis.py                 # Diferencias en Diferencias: episodios PM2.5 → mortalidad respiratoria
│   │   ├── iv_analysis.py                  # Variables Instrumentales: altura de capa límite como instrumento para PM2.5
│   │   └── heterogeneous_effects.py        # Causal Forest (EconML): CATE por región, estación y grupo de edad
│   ├── model/
│   │   ├── train.py                        # entrena clasificadores LightGBM y XGBoost; registra ejecuciones en MLflow
│   │   └── evaluate.py                     # ajuste de umbral, curvas recall/precisión, valores SHAP, desglose por país
│   └── pipeline/
│       └── flow.py                         # DAG de Prefect — conecta todas las tareas, gestiona flags, programa reentrenamiento semanal
├── api/
│   └── main.py                             # app FastAPI — endpoints /predict, /predict/batch, /health-impact, /metrics
├── dashboard/
│   └── app.py                              # Plotly Dash — tendencias PM2.5 en Europa, mapas de mortalidad, rendimiento del modelo
├── tests/
│   └── test_api.py                         # tests de integración de la API, ejecutados en cada PR via GitHub Actions
├── models/                             # artefactos del modelo guardados (excluidos de git)
├── Dockerfile                          # imagen de producción para la API
├── Dockerfile.pipeline                 # imagen del pipeline (Spark + Prefect)
├── docker-compose.yml
├── prefect.yaml
├── requirements-ingestion.txt
├── requirements-spark.txt
├── requirements-model.txt
├── requirements-causal.txt
├── requirements-api.txt
├── requirements-dashboard.txt
└── requirements-pipeline.txt
```

---

### Inicio Rápido

```bash
# 1. Instalar dependencias de ingesta
pip install -r requirements-ingestion.txt

# 2. Configurar acceso a ERA5 (configuración única)
# Regístrate en https://cds.climate.copernicus.eu y guarda tu clave en ~/.cdsapirc

# 3. Descargar datos (primero en modo dry-run para comprobar tamaños)
python src/ingestion/download_eea.py --dry-run
python src/ingestion/download_era5.py --dry-run
python src/ingestion/download_eurostat.py --dry-run

# 4. Descarga real
python src/ingestion/download_eea.py
python src/ingestion/download_era5.py
python src/ingestion/download_eurostat.py
python src/ingestion/download_who.py

# 5. Validar descargas
python src/ingestion/validate_downloads.py

# 6. Ejecutar jobs de Spark
pip install -r requirements-spark.txt
spark-submit src/spark/spark_clean_eea.py
spark-submit src/spark/spark_clean_era5.py
spark-submit src/spark/spark_clean_health.py
spark-submit src/spark/spark_join_features.py

# 7. Entrenar modelos
pip install -r requirements-model.txt
python src/model/train.py
python src/model/evaluate.py

# 8. Ejecutar análisis causales (ver notebooks/ para documentación narrativa)
pip install -r requirements-causal.txt
jupyter notebook notebooks/03_causal_did.ipynb

# 9. Ejecutar la API en local
pip install -r requirements-api.txt
uvicorn api.main:app --reload
# Swagger UI → http://localhost:8000/docs

# 10. Ejecutar el dashboard
pip install -r requirements-dashboard.txt
python dashboard/app.py
# Dashboard → http://localhost:8050
```

---

*Project by — open to contributions and feedback.*
