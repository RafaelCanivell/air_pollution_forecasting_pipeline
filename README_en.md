# Air Quality & Public Health Impact Pipeline
### EEA · ERA5 · EUROSTAT — France, Spain, Belgium, Netherlands, Germany (2019–2023)

> **Work in progress.** Active development — some stages and docs are incomplete.

> **Engineering philosophy.** Built with a deliberate focus on MLOps principles: reliable, reusable, maintainable, flexible, and fully reproducible.

[![CI](https://github.com/your-org/eea-era5-air-quality/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/eea-era5-air-quality/actions)

---

## What this does

An end-to-end ML and causal inference pipeline that:

1. **Predicts** dangerous PM2.5 exceedance events 24–48 hours in advance (LightGBM / XGBoost, recall ≥ 90 %)
2. **Estimates** the causal effect of those events on respiratory mortality and cardiovascular hospitalisations (DiD, IV, Causal Forest)
3. **Serves** both outputs through a production-grade REST API

```
EEA + ERA5 + EUROSTAT
        │
        ▼
  Feature store (Parquet)
        │
   ┌────┴────┐
   │         │
   ▼         ▼
Prediction  Causal
  flow       flow
 (alert)  (policy estimate)
   │         │
   └────┬────┘
        ▼
    REST API
 /predict · /health-impact · /metrics
```

---

## Who should read what

| You are… | Start here |
|---|---|
| ML / Data Engineer | This README → [docs/METHODOLOGY.md](docs/METHODOLOGY.md) → [docs/MONITORING.md](docs/MONITORING.md) |
| Epidemiologist / Researcher | [docs/CAUSAL_INFERENCE.md](docs/CAUSAL_INFERENCE.md) → [docs/CAUSAL_DAG.md](docs/CAUSAL_DAG.md) |
| Policy analyst | [docs/TARGET_AUDIENCE.md](docs/TARGET_AUDIENCE.md) → API endpoints below |
| New contributor | [docs/METHODOLOGY.md](docs/METHODOLOGY.md) → [docs/DAG_PREFECT.md](docs/DAG_PREFECT.md) |

---

## Prerequisites

- Python ≥ 3.10, Java 11+ (for Spark), Docker
- [Copernicus CDS account](https://cds.climate.copernicus.eu) — free, needed for ERA5
- ~25 GB disk for raw + processed data (5 countries, 5 years)

---

## Quick Start

```bash
# 1. Clone and install ingestion deps
git clone https://github.com/your-org/eea-era5-air-quality.git
pip install -r requirements-ingestion.txt

# 2. Configure ERA5 access (one-time)
# Register at https://cds.climate.copernicus.eu -> save key to ~/.cdsapirc

# 3. Dry-run to check download sizes
python src/ingestion/download_eea.py --dry-run
python src/ingestion/download_era5.py --dry-run

# 4. Full pipeline via Makefile
make data      # download + validate + Spark processing
make train     # feature engineering + model training
make causal    # run DiD, IV, Causal Forest notebooks
make serve     # launch FastAPI on localhost:8000
```

Local mode (no Spark required):
```bash
spark-submit src/spark/spark_clean_eea.py --engine pandas
```

---

## API endpoints

Base URL: `https://your-app.onrender.com` · [Swagger UI](http://localhost:8000/docs)

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | PM2.5 exceedance probability at t+1 and t+2 |
| `/predict/batch` | POST | Same for multiple stations |
| `/health-impact` | GET | Causal estimate of additional deaths / admissions |
| `/metrics` | GET | Live recall, precision, drift scores |
| `/health` | GET | Service health check |

**Example `/predict` request:**
```json
{
  "station_id": "FR04012",
  "timestamp": "2024-01-15",
  "features": {
    "pm10_value": 45.2, "no2_value": 38.1, "o3_value": 52.3,
    "so2_value": 3.4, "temperature": 2.5, "wind_speed": 4.1,
    "boundary_layer_height": 320
  }
}
```

---

## Documentation

| Document | Contents |
|---|---|
| [TARGET_AUDIENCE.md](docs/TARGET_AUDIENCE.md) | Who this is for and how each audience should engage |
| [METHODOLOGY.md](docs/METHODOLOGY.md) | Phases 1–6, design decisions, modelling choices, tech stack |
| [CAUSAL_INFERENCE.md](docs/CAUSAL_INFERENCE.md) | DiD, IV, Causal Forest — methods, assumptions, validity checks |
| [CAUSAL_DAG.md](docs/CAUSAL_DAG.md) | Directed Acyclic Graph — variable relationships, blocked paths |
| [MONITORING.md](docs/MONITORING.md) | Offline metrics, production drift detection, retraining logic |
| [DAG_PREFECT.md](docs/DAG_PREFECT.md) | Prefect orchestration — task graph, flag gates, scheduling |
| [LIMITATIONS.md](docs/LIMITATIONS.md) | Known limitations — read before interpreting results |

---

## Data sources

| Source | Dataset |
|---|---|
| EEA | [Air Quality e-Reporting](https://www.eea.europa.eu/en/datahub/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d) |
| Copernicus/ECMWF | [ERA5 Reanalysis](https://cds.climate.copernicus.eu) |
| EUROSTAT | `demo_r_mweek3`, `hlth_cd_aro`, `hlth_co_hospit` |
| WHO Europa | [European Mortality Database](https://gateway.euro.who.int/en/datasets/european-mortality-database) |

---

*Open to contributions and feedback.*
