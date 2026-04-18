# Prefect DAG — Pipeline Orchestration

This document describes the Prefect orchestration layer: the task graph, dependency mechanism, scheduling, and task design principles. For the causal variable DAG, see [`CAUSAL_DAG.md`](CAUSAL_DAG.md).

---

## Two DAGs in this project — do not confuse them

| DAG | What it represents | Document |
|---|---|---|
| **Prefect task graph** | The execution order of pipeline steps (code) | This file |
| **Causal DAG** | The assumed causal relationships between variables (science) | [`CAUSAL_DAG.md`](CAUSAL_DAG.md) |

---

## Full task graph

```
flow: weekly_air_quality_pipeline
schedule: every Monday 02:00 UTC
│
├── [INGESTION]  ── runs in parallel ──────────────────────────────────
│   ├── download_eea()
│   │       └── validate_eea()          ──► writes flag: data/flags/eea.ok
│   ├── download_era5()
│   │       └── validate_era5()         ──► writes flag: data/flags/era5.ok
│   └── download_health()
│           └── validate_health()       ──► writes flag: data/flags/health.ok
│
├── [PROCESSING]  ── gates on ALL three *.ok flags ───────────────────
│   ├── spark_clean_eea()
│   ├── spark_clean_era5()
│   ├── spark_clean_health()
│   └── spark_join_features()           ──► writes flag: data/flags/features.ok
│
├── [PREDICTION BRANCH]  ── gates on features.ok ─────────────────────
│   ├── train_lgbm()                    ──► logs to MLflow run A
│   ├── train_xgboost()                 ──► logs to MLflow run B
│   ├── evaluate_all_models()           ── compares recall, PSI, threshold
│   ├── select_champion()               ── gates on recall>=0.90 AND PSI<=0.25
│   └── deploy_api()                    ──► writes flag: data/flags/model.deployed
│           └── smoke_test_api()        ── POST /predict, GET /health
│
├── [CAUSAL BRANCH]  ── gates on features.ok, PARALLEL to prediction ─
│   ├── run_did_analysis()              ──► writes data/processed/causal/did_results.parquet
│   ├── run_iv_analysis()               ──► writes data/processed/causal/iv_results.parquet
│   └── run_causal_forest()             ──► writes data/processed/causal/cate_maps.parquet
│
├── [MONITORING]  ── gates on model.deployed ─────────────────────────
│   ├── compute_drift_metrics()         ── PSI per feature, KS test, KL divergence
│   ├── compute_rolling_recall()        ── requires EEA ground truth (24-48h lag)
│   ├── evaluate_retrain_trigger()      ── see MONITORING.md for logic
│   └── alert_if_degraded()             ── Prefect notification + MLflow tag
│
└── [ARTEFACTS]  ── runs last, always ───────────────────────────────
    └── log_monitoring_to_mlflow()      ── separate monitoring run in MLflow
```

---

## Flag-file dependency mechanism

Tasks do not depend on each other through Prefect's built-in `wait_for` alone. A flag file written to `data/flags/` acts as a secondary gate — a downstream task checks for the flag's existence before starting and raises `SKIP` if it is absent.

This mechanism provides two benefits:
1. **Incremental execution.** If the Prefect flow is re-run after a partial failure, tasks whose flags already exist are skipped — only failed tasks and their dependents re-execute.
2. **Cross-process safety.** Flag files persist across Prefect agent restarts. A task that completed in a previous run will not re-execute even if the agent was restarted.

```python
# Pattern used in all gated tasks
from pathlib import Path
from prefect import task, get_run_logger

FLAGS_DIR = Path("data/flags")

def require_flag(flag_name: str) -> None:
    flag_path = FLAGS_DIR / flag_name
    if not flag_path.exists():
        raise RuntimeError(
            f"Required flag '{flag_name}' not found. "
            f"Upstream task may have failed or been skipped."
        )

def write_flag(flag_name: str) -> None:
    FLAGS_DIR.mkdir(parents=True, exist_ok=True)
    (FLAGS_DIR / flag_name).touch()

@task(retries=3, retry_delay_seconds=60)
def spark_join_features() -> None:
    logger = get_run_logger()
    require_flag("eea.ok")
    require_flag("era5.ok")
    require_flag("health.ok")
    logger.info("All upstream flags present — starting feature join")
    # ... spark logic ...
    write_flag("features.ok")
    logger.info("Feature store complete — flag written")
```

---

## Task design principles

Every task in the flow is designed to satisfy the following constraints. These are not guidelines — they are enforced through code review.

**1. Pure function with typed inputs and outputs.**
Tasks receive their inputs as parameters and return their outputs explicitly. No task reads from or writes to shared mutable state outside the flag-file and Parquet conventions above.

```python
# Good
@task
def train_lgbm(feature_store_path: Path, config: dict) -> Path:
    ...
    return model_artefact_path

# Bad — reads from a global or mutates external state directly
@task
def train_lgbm():
    df = pd.read_parquet(GLOBAL_FEATURE_STORE)  # implicit dependency
    MODEL_REGISTRY.register(model)              # side effect, not in signature
```

**2. Retry only on network I/O tasks.**
`@task(retries=3, retry_delay_seconds=60)` is applied exclusively to tasks that perform network calls (download, API deployment, MLflow logging). Processing and modelling tasks are not retried — a failure there indicates a code or data problem that should not be retried blindly.

**3. Idempotent by design.**
Every task must produce the same result when run multiple times on the same input. Spark jobs overwrite their output Parquet partitions. Model training is fixed via seeds. Flag files are idempotent (`.touch()` on an existing file is a no-op).

**4. Logging at entry and exit.**
Every task logs its start (with key input parameters) and its successful completion (with key output summary statistics). This makes the Prefect UI actionable for debugging without reading the code.

```python
@task
def validate_eea(raw_path: Path) -> None:
    logger = get_run_logger()
    logger.info(f"Validating EEA data at {raw_path}")
    # ... validation logic ...
    logger.info(f"EEA validation passed: {n_files} files, {total_rows:,} rows, {n_stations} stations")
    write_flag("eea.ok")
```

**5. The prediction and causal branches run in parallel.**
After `features.ok` is set, both branches start concurrently. There is no dependency between `train_lgbm()` and `run_did_analysis()`. This is enforced by Prefect's task graph — neither branch calls any task from the other.

---

## Scheduling

```yaml
# prefect.yaml
deployments:
  - name: weekly-pipeline
    schedule:
      cron: "0 2 * * 1"   # Every Monday at 02:00 UTC
      timezone: "UTC"
    parameters:
      engine: spark        # use --engine pandas for local dev
      countries: [FR, ES, BE, NL, DE]
      years: [2019, 2020, 2021, 2022, 2023]
      retrain_if_triggered: true
```

The schedule is chosen for Monday 02:00 UTC because:
- EEA data for the previous week is typically finalised by Sunday evening
- EUROSTAT weekly health data updates on Monday mornings
- 02:00 UTC avoids peak API load on Render.com

---

## Running locally

```bash
# Full flow (requires Spark and Prefect agent)
prefect deployment run weekly-pipeline/default

# Single task (for debugging)
python -c "
from src.pipeline.flow import spark_join_features
spark_join_features(engine='pandas')
"

# Dry-run without Prefect (pure Python)
python src/spark/spark_join_features.py --engine pandas --dry-run
```

---

## Failure modes and recovery

| Failure | Symptom | Recovery |
|---|---|---|
| EEA download timeout | `eea.ok` flag not written | Re-run flow — download task retries 3× before raising |
| ERA5 CDS quota exceeded | `validate_era5` raises quota error | Wait 24h, re-run. CDS has daily download limits. |
| Spark OOM | `spark_join_features` fails | Reduce `--partitions` parameter or use `--engine pandas` |
| Champion selection fails | New model does not meet recall threshold | Current production model is retained; alert raised; manual investigation required |
| Render.com deploy fails | `model.deployed` flag not written | `smoke_test_api` will not run; monitoring will skip; manual deploy via `make serve` |

All failure events are logged to MLflow as a tag on the affected run and surfaced in the Prefect UI. No silent failures — every task either succeeds, retries, or raises visibly.
