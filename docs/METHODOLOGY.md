# Methodology

Full description of the six pipeline phases, modelling design decisions, and tech stack.

---

## Phase 1 — Ingestion and Exploratory Analysis

Raw data is downloaded programmatically from four sources:

- **EEA Air Quality Download Service** — PM2.5, PM10, NO₂, O₃, SO₂ Parquet files per country and year, via the `airbase` Python package
- **Copernicus Climate Data Store (ERA5)** — monthly NetCDF files for temperature, wind speed, humidity, atmospheric pressure, and boundary layer height, via `cdsapi`
- **EUROSTAT** — three health outcome tables (`demo_r_mweek3`, `hlth_cd_aro`, `hlth_co_hospit`) via the `eurostat` Python package
- **WHO European Mortality Database** — ZIP archive of weekly mortality CSVs by cause, age, and country

All download scripts (`src/ingestion/`) support `--dry-run` to inspect sizes before committing to a full download. Retry logic with exponential backoff (`src/utils/retry.py`) handles transient failures from the EEA and CDS endpoints.

**EDA scope (`notebooks/01_eda.ipynb`):** data coverage and gap patterns, missing value rates by station and country, PM2.5 distribution shape and class imbalance (exceedance events are ~8–12 % of station-days depending on country and season), cross-pollutant correlations, temporal autocorrelation of PM2.5, seasonal structure, and mortality trends by cause, age, and country.

---

## Phase 2 — Large-Scale Processing with Apache Spark

PySpark jobs (`src/spark/`) handle four tasks in sequence, gated by flag files:

| Job | Input | Output | Key operations |
|---|---|---|---|
| `spark_clean_eea.py` | Raw EEA Parquet | Cleaned EEA Parquet, partitioned by country+year | Deduplication, outlier capping, schema enforcement |
| `spark_clean_era5.py` | Raw NetCDF | ERA5 Parquet at EEA station coordinates | NetCDF→Parquet via `xarray`, spatial interpolation (inverse-distance weighting to EEA station coords) |
| `spark_clean_health.py` | EUROSTAT + WHO CSVs | Harmonised health Parquet at NUTS3-week level | NUTS3 code harmonisation across vintages, cause-of-death mapping |
| `spark_join_features.py` | All three above | Feature store Parquet | Station→NUTS3 spatial join, lag features (t-1, t-2), rolling windows (7-day, 14-day), calendar features, binary target label |

**Why Spark at ~2M rows:** pandas is technically sufficient. Spark is retained to (1) demonstrate production-grade data engineering and (2) ensure the pipeline scales to additional countries or years with only a configuration change. All jobs accept `--engine pandas` for local development without Spark overhead.

**Training-serving skew prevention:** all feature transformations are implemented as pure functions in `src/utils/transforms.py`, imported by both the Spark jobs and the FastAPI inference layer. There is no separate preprocessing logic in the API.

---

## Phase 3 — Orchestration with Prefect

All steps are wrapped in a Prefect flow (`src/pipeline/flow.py`) with:

- Explicit task dependencies via flag files (`data/flags/*.flag`) — a downstream task will not start if the upstream flag does not exist
- Schema validation gates after each ingestion step
- Retry logic (`@task(retries=3, retry_delay_seconds=60)`) on all network I/O
- Incremental execution — only new data is downloaded on each weekly run
- Conditional retraining — a model retrain is triggered only when new data arrived AND either rolling recall < 80 % or max PSI > 0.25

See [`DAG_PREFECT.md`](DAG_PREFECT.md) for the full task graph.

---

## Phase 4 — Predictive Modelling

**Task:** binary classification — PM2.5 > 25 µg/m³ (EU daily limit) at t+1 (24h) and t+2 (48h).

**Models trained:**

| Model | Role | Library |
|---|---|---|
| LightGBM | Primary — best recall/speed trade-off | `lightgbm` |
| XGBoost | Comparative — same features, different boosting | `xgboost` |
| Logistic Regression | Interpretable baseline | `sklearn` |

**Key design decisions:**

| Decision | Choice | Rationale |
|---|---|---|
| Primary metric | Recall ≥ 90 % | A missed danger event has higher cost than a false alarm |
| Decision threshold | Tuned per model on validation set | Never fixed at 0.5; stored as model artefact |
| Validation strategy | Strict temporal split | Random splits cause leakage via lag features |
| Lag feature computation | Training data only within each fold | No future information leaks into past folds |
| Class imbalance | `scale_pos_weight` in LightGBM/XGBoost | Danger days are ~10 % of observations |
| Evaluation breakdown | Per-country recall | Identifies regional model weaknesses before deployment |
| Interpretability | SHAP values | Per-prediction feature attribution, stored as model artefact |

**Threshold tuning procedure:** for each model, the decision threshold is swept from 0.3 to 0.7 in steps of 0.01. The lowest threshold that achieves recall ≥ 90 % on the validation set is selected and persisted alongside the model binary in MLflow.

**Champion selection (`select_champion` task):** a newly trained model is registered and deployed only if its validation recall meets threshold AND its PSI drift score is within bounds relative to the current production model. This prevents deploying a model that performs well on validation but was trained on a data distribution that has drifted from production.

---

## Phase 5 — Causal Inference

Three analyses estimating the health burden of PM2.5 exceedance events. These run in parallel with the predictive modelling branch after the feature store is ready.

See [`CAUSAL_INFERENCE.md`](CAUSAL_INFERENCE.md) for full methodology, and [`CAUSAL_DAG.md`](CAUSAL_DAG.md) for the assumed causal structure.

**Summary:**

| Analysis | Method | Outcome | Library |
|---|---|---|---|
| DiD | Difference-in-Differences | Weekly respiratory deaths | `linearmodels` |
| IV | Instrumental Variables (BLH instrument) | Cardiovascular hospital admissions | `linearmodels` |
| Causal Forest | Heterogeneous treatment effects | CATE by NUTS3, season, age group | `econml` |

---

## Phase 6 — Production Deployment

**API:** FastAPI application (`api/main.py`) containerised with Docker (multi-stage build, non-root user, health check endpoint). Exposes `/predict`, `/predict/batch`, `/health-impact`, `/metrics`, `/health`.

**CI/CD:** GitHub Actions runs `tests/test_api.py` on every PR. Push to `main` triggers automatic deploy to Render.com.

**Monitoring:** see [`MONITORING.md`](MONITORING.md) for the full monitoring design.

---

## Tech stack

| Layer | Tool |
|---|---|
| Ingestion — air quality | Python, `airbase` |
| Ingestion — meteorology | Python, `cdsapi` (ERA5 / Copernicus) |
| Ingestion — health | Python, `eurostat` package, `requests` (WHO) |
| Processing | Apache Spark (PySpark), `xarray` (NetCDF→Parquet); `--engine pandas` flag for local dev |
| Feature engineering | PySpark + pure-function transforms shared with API layer |
| Orchestration | Prefect — flag gates, retry logic, incremental execution |
| Experiment tracking | MLflow — hyperparameters, metrics, thresholds, SHAP artefacts |
| Modelling | LightGBM, XGBoost, scikit-learn, SHAP |
| Causal inference | EconML, DoWhy, `linearmodels` |
| Visualisation | Plotly Dash |
| API | FastAPI |
| Containerisation | Docker (multi-stage, non-root) |
| CI/CD | GitHub Actions |
| Deploy | Render.com (free tier — see limitations) |
| Monitoring | Custom: PSI, KS test, KL divergence, rolling recall |

---

## Geographic scope

Five countries: France, Spain, Belgium, the Netherlands, Germany — 2019–2023.

| Criterion | Single country | 5-country scope | All Europe |
|---|---|---|---|
| EEA stations | ~70 | ~1,000–1,200 | ~5,000+ |
| Daily rows (5 years) | ~130K | ~2M | ~100M+ |
| Pandas sufficient? | Yes | Borderline | No |
| Spark justified? | No | Yes (demonstrable + scalable) | Yes (necessary) |
| DiD control groups | Few | Rich | Rich |
| Transboundary signal | No | Yes | Yes |

The five-country bloc was chosen for its documented transboundary pollution interactions (Rhine-Ruhr industrial emissions affecting Belgium and the Netherlands; Saharan dust episodes affecting Spain and France simultaneously) — exactly the cross-border variation the DiD and IV analyses need.
