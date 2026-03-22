# Air Quality & Public Health Impact Pipeline — EEA + ERA5 + EUROSTAT
### *Predicting dangerous pollution events and estimating their causal effect on respiratory & cardiovascular mortality across Europe*

---

> **English version below** | **Versión en español más abajo**

---

## English

### Business Case & Problem Statement

Air pollution is the single largest environmental health risk in Europe, responsible for over **400,000 premature deaths per year** across the EU according to the European Environment Agency. Despite significant progress in reducing emissions over recent decades, pollutant concentrations still exceed both EU legal limits and the stricter WHO guidelines in many urban and industrial areas — particularly in Central and Eastern Europe, the Po Valley in Italy, and heavily trafficked Western European cities.

The economic cost is equally severe: lost working days, hospital admissions, and long-term chronic disease place hundreds of billions of euros of pressure on European healthcare systems annually, with the burden falling disproportionately on lower-income populations living near industrial zones, highways, and ports.

Despite the availability of dense monitoring networks, most public-facing tools offer only *current* air quality readings — they do not **predict** when conditions will become dangerous, nor do they **quantify** the downstream health burden those conditions cause. A 24 to 48-hour advance warning of a high-pollution event allows individuals, hospitals, and municipal governments to take preventive action: issuing health advisories, adjusting industrial operations, activating low-emission zones, or simply advising vulnerable populations to stay indoors.

This project addresses both gaps by building an **end-to-end machine learning and causal inference pipeline** that ingests historical atmospheric, pollution, and health data across Europe, engineers predictive features at scale, estimates the causal effect of pollution on mortality and hospitalizations, and serves real-time forecasts through a production-grade API.

---

### Why Combine EEA, ERA5, and Health Data?

These three data sources are individually powerful but tell incomplete stories on their own.

**EEA Air Quality e-Reporting** provides ground-level measurements of five key pollutants — PM2.5, PM10, ozone (O₃), nitrogen dioxide (NO₂), and sulphur dioxide (SO₂) — collected from thousands of official monitoring stations across all EU member states and several neighboring countries. It tells us *what the air contains*, but not *why* conditions changed, *what will happen next*, or *how many people will be harmed*.

**ERA5 (Copernicus/ECMWF)** is the European reference for atmospheric reanalysis, providing continuous gridded data — temperature, wind speed, humidity, atmospheric pressure, and boundary layer height — at 0.25° resolution from 1940 to present. It tells us about the *atmospheric conditions* that drive pollutant behavior: temperature inversions that trap particles near the ground, wind patterns that disperse or concentrate pollutants across national borders, and precipitation events that temporarily clean the air. As a 100% European dataset produced by the European Centre for Medium-Range Weather Forecasts, it replaces NOAA GHCN entirely and offers superior spatial coverage through a uniform grid rather than sparse station networks.

**EUROSTAT and WHO European Mortality Database** provide the health outcome layer: weekly mortality by cause, age, and NUTS3 region, and hospital admissions for respiratory and cardiovascular conditions. They tell us *how many people were harmed* — the outcome that gives the entire pipeline its public health meaning.

Combining all three allows the pipeline to both predict dangerous pollution events and rigorously estimate their causal downstream effects on human health.

The transboundary nature of European air pollution — where industrial emissions from one country affect air quality in neighboring ones — makes the multi-country, multi-source approach especially relevant. This is also what justifies Apache Spark: the combined historical dataset spans decades, dozens of countries, and hundreds of millions of records that do not fit comfortably in memory.

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
| Modeling | LightGBM, XGBoost, scikit-learn, SHAP |
| Causal inference | EconML, DoWhy, `linearmodels` |
| Visualization | Plotly Dash |
| API | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deploy | Render.com (free tier) |

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
│   ├── ingestion/
│   │   ├── download_eea.py
│   │   ├── download_era5.py
│   │   ├── download_eurostat.py
│   │   ├── download_who.py
│   │   └── validate_downloads.py
│   ├── spark/
│   │   ├── spark_clean_eea.py
│   │   ├── spark_clean_era5.py
│   │   ├── spark_clean_health.py
│   │   └── spark_join_features.py
│   ├── causal/
│   │   ├── did_analysis.py
│   │   ├── iv_analysis.py
│   │   └── heterogeneous_effects.py
│   ├── model/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── pipeline/
│       └── flow.py                     # Prefect DAG
├── api/
│   └── main.py                         # FastAPI application
├── dashboard/
│   └── app.py                          # Plotly Dash dashboard
├── tests/
│   └── test_api.py
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
| 🇪🇺 EEA | Air Quality e-Reporting | https://www.eea.europa.eu/en/datahub/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d |
| 🇪🇺 Copernicus/ECMWF | ERA5 Reanalysis | https://cds.climate.copernicus.eu |
| 🇪🇺 EUROSTAT | Weekly mortality by NUTS3 (`demo_r_mweek3`) | https://ec.europa.eu/eurostat |
| 🇪🇺 EUROSTAT | Deaths by respiratory/cardiovascular cause (`hlth_cd_aro`) | https://ec.europa.eu/eurostat |
| 🇪🇺 EUROSTAT | Hospital admissions (`hlth_co_hospit`) | https://ec.europa.eu/eurostat |
| 🇪🇺 WHO Europa | European Mortality Database | https://gateway.euro.who.int/en/datasets/european-mortality-database |

---
---

## Español

### (Trabajo en curso) Caso de Negocio y Planteamiento del Problema

La contaminación del aire es el mayor riesgo ambiental para la salud en Europa, responsable de más de **400.000 muertes prematuras al año** en la UE según la Agencia Europea de Medio Ambiente. A pesar del progreso significativo en la reducción de emisiones, las concentraciones de contaminantes siguen superando tanto los límites legales de la UE como las directrices más estrictas de la OMS en muchas zonas urbanas e industriales — especialmente en Europa Central y del Este, el Valle del Po en Italia y las ciudades del oeste con alto tráfico.

La mayoría de las herramientas públicas solo ofrecen lecturas *actuales* de calidad del aire — no **predicen** cuándo las condiciones se volverán peligrosas, ni **cuantifican** la carga sanitaria que esas condiciones generan. Un aviso de 24 a 48 horas antes de un evento de alta contaminación permite a personas, hospitales y gobiernos municipales tomar medidas preventivas: emitir alertas sanitarias, ajustar operaciones industriales, activar zonas de bajas emisiones, o simplemente indicar a poblaciones vulnerables que permanezcan en interiores.

Este proyecto aborda ambas brechas construyendo un **pipeline end-to-end de machine learning e inferencia causal** que ingesta datos históricos atmosféricos, de contaminación y de salud a escala europea, genera features predictivas a gran escala, estima el efecto causal de la contaminación sobre la mortalidad y hospitalizaciones, y sirve predicciones en tiempo real a través de una API lista para producción.

---

### ¿Por Qué Combinar EEA, ERA5 y Datos de Salud?

**EEA Air Quality e-Reporting** nos dice *qué contiene el aire* — niveles de PM2.5, PM10, O₃, NO₂ y SO₂ — pero no *por qué cambiaron las condiciones*, *qué pasará después*, ni *cuántas personas serán dañadas*.

**ERA5 (Copernicus/ECMWF)** es la referencia europea para el reanálisis atmosférico, proporcionando datos en grid continuo — temperatura, velocidad del viento, humedad, presión y altura de la capa límite — a resolución de 0,25° desde 1940. Nos habla de las *condiciones atmosféricas* que determinan el comportamiento de los contaminantes: inversiones térmicas que atrapan partículas cerca del suelo, patrones de viento que dispersan o concentran contaminantes a través de fronteras, y precipitaciones que limpian temporalmente el aire. Como dataset 100% europeo producido por el Centro Europeo de Predicción Meteorológica a Plazo Medio, reemplaza completamente a NOAA GHCN y ofrece cobertura espacial superior mediante un grid uniforme.

**EUROSTAT y la Base de Datos Europea de Mortalidad de la OMS** aportan la capa de resultados sanitarios: mortalidad semanal por causa, edad y región NUTS3, y hospitalizaciones por enfermedades respiratorias y cardiovasculares. Nos dicen *cuántas personas fueron dañadas* — el resultado que otorga al pipeline todo su significado de salud pública.

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
| Modelado | LightGBM, XGBoost, scikit-learn, SHAP |
| Inferencia causal | EconML, DoWhy, `linearmodels` |
| Visualización | Plotly Dash |
| API | FastAPI |
| Containerización | Docker |
| CI/CD | GitHub Actions |
| Deploy | Render.com (free tier) |

---

### Fuentes de Datos

| Fuente | Dataset | URL |
|---|---|---|
| 🇪🇺 EEA | Air Quality e-Reporting | https://www.eea.europa.eu/en/datahub/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d |
| 🇪🇺 Copernicus/ECMWF | ERA5 Reanalysis | https://cds.climate.copernicus.eu |
| 🇪🇺 EUROSTAT | Mortalidad semanal NUTS3 (`demo_r_mweek3`) | https://ec.europa.eu/eurostat |
| 🇪🇺 EUROSTAT | Muertes por causa respiratoria/cardiovascular (`hlth_cd_aro`) | https://ec.europa.eu/eurostat |
| 🇪🇺 EUROSTAT | Hospitalizaciones (`hlth_co_hospit`) | https://ec.europa.eu/eurostat |
| 🇪🇺 WHO Europa | European Mortality Database | https://gateway.euro.who.int/en/datasets/european-mortality-database |

---

*Project by — open to contributions and feedback.*
