# Target Audience

This document describes the three primary audiences for this project and explains what each will find most useful — and where to start.

---

## Primary audiences

### 1. ML Engineers and Data Scientists

**What you will find here:** A production-grade MLOps pipeline demonstrating the full lifecycle from raw data ingestion to containerised deployment and monitoring. The codebase is structured for readability, modularity, and reproducibility — not just as a working prototype.

**Specifically relevant:**
- Strict temporal train/val/test splitting to prevent data leakage through lag features
- MLflow experiment tracking with full artefact logging (hyperparameters, metrics, SHAP values, model binaries)
- Prefect DAG orchestration with flag-file dependency gates and exponential-backoff retry logic
- Decision threshold tuning for imbalanced classification (recall ≥ 90 % as primary objective)
- PSI, KS, and KL-divergence drift detection with retraining triggers
- Docker multi-stage builds, non-root user, GitHub Actions CI/CD to Render.com

**Suggested reading order:**
1. [`METHODOLOGY.md`](METHODOLOGY.md) — engineering phases and design decisions
2. [`MONITORING.md`](MONITORING.md) — offline metrics, production monitoring, drift detection
3. [`DAG_PREFECT.md`](DAG_PREFECT.md) — orchestration and task design principles
4. `src/pipeline/flow.py` — the actual Prefect flow

---

### 2. Public Health Researchers and Epidemiologists

**What you will find here:** Three embedded causal analyses grounded in established econometric methodology, using high-quality European administrative health data (EUROSTAT, WHO European Mortality Database). The causal design choices — instrument selection, identification assumptions, robustness checks — are documented in full narrative in the notebooks.

No ML background is required to follow or reuse the causal analyses independently of the prediction pipeline. The causal modules (`src/causal/`) and their corresponding notebooks (`notebooks/03–05`) are self-contained.

**Specifically relevant:**
- Difference-in-Differences with ERA5 meteorological covariates as time-varying controls
- Instrumental Variables using boundary layer height — instrument validity formally verified (first-stage F > 10, Sargan test)
- Causal Forest (EconML) for heterogeneous treatment effect estimation by NUTS3 region, season, and age group
- Explicit DAG documenting all assumed causal relationships and blocked paths

**Suggested reading order:**
1. [`CAUSAL_DAG.md`](CAUSAL_DAG.md) — start here to understand the assumed causal structure
2. [`CAUSAL_INFERENCE.md`](CAUSAL_INFERENCE.md) — methods, assumptions, and validity checks for each analysis
3. `notebooks/03_causal_did.ipynb` — DiD with full narrative
4. `notebooks/04_causal_iv.ipynb` — IV with instrument validation
5. `notebooks/05_causal_heterogeneity.ipynb` — Causal Forest CATE maps

**Known limitation for this audience:** The causal estimates are retrospective averages over 2019–2023. They quantify the historical average treatment effect — they are not forward-looking causal forecasts. See [`LIMITATIONS.md`](LIMITATIONS.md) for the full list.

---

### 3. Policy Analysts and Municipal Decision-Makers

**What you will find here:** A reference architecture for a real-time early warning system. The `/health-impact` API endpoint translates a pollution forecast directly into estimated mortality and hospitalisation burden — a direct input for cost-benefit analyses of emission regulations, low-emission zone decisions, or health advisory protocols.

**Specifically relevant:**
- The `/predict` endpoint provides 24–48h advance warning of PM2.5 exceedance events per monitoring station
- The `/health-impact` endpoint returns causal estimates of additional respiratory deaths and cardiovascular hospital admissions associated with an episode of a given magnitude
- The dashboard (`dashboard/app.py`) provides a visual interface for Europe-wide PM2.5 trends and regional health burden maps

**What this project does not do:**
- It does not provide real-time streaming data (weekly batch only — see [`LIMITATIONS.md`](LIMITATIONS.md))
- The causal estimates are statistical averages, not deterministic predictions for a specific episode
- The deployment on Render.com free tier has cold-start latency and no horizontal scaling — suitable for demonstration, not for production workloads with SLA requirements

**Suggested reading order:**
1. [`CAUSAL_INFERENCE.md`](CAUSAL_INFERENCE.md) — section on "What causal estimates mean operationally"
2. [`LIMITATIONS.md`](LIMITATIONS.md) — mandatory before citing any output in policy documents
3. The main README for API endpoint documentation

---

## Secondary audiences

**MSc / PhD students** looking for a real-world case study combining supervised ML and causal inference on environmental data. The project intentionally exposes design trade-offs (why Spark at 2M rows, why BLH as instrument, why recall over precision) with written justification rather than just presenting final choices.

**Data engineers** interested in the Spark + Prefect + MLflow integration pattern, the flag-file dependency gate mechanism, and the training-serving skew prevention strategy (shared preprocessing code between `src/spark/` and `api/main.py`).
