"""
flow.py
-------
Prefect flow that orchestrates the full pipeline from data ingestion
to model training, running on a weekly schedule.

What is Prefect?
----------------
Prefect is a workflow orchestration tool. You define:
  - @task: a single unit of work (a function)
  - @flow: a collection of tasks with dependencies between them

Prefect handles:
  - Execution order and dependency resolution
  - Retry logic and failure handling at the task level
  - Scheduling (cron or interval)
  - Logging and observability via the Prefect UI
  - State management (which tasks ran, which failed, which were skipped)

Relationship to the ingestion scripts and Spark jobs
-----------------------------------------------------
The ingestion scripts (download_*.py) and Spark jobs (spark_clean_*.py)
are written as plain Python functions with no Prefect dependency.
This is intentional: they can be run standalone, tested independently,
and ported to a different orchestrator without changes.

flow.py simply imports those functions and wraps them in @task decorators.
The @task wrapper adds retry logic, state tracking, and logging on top of
the existing function — it does not change what the function does.

Relationship to MLflow
----------------------
MLflow is called inside train_models() — specifically inside src/model/train.py
which is called here. Prefect does not know about MLflow; it only sees that
train_models() succeeded or failed. MLflow tracks what happened inside that call.

Pipeline structure
------------------

[download_eea]  [download_era5]  [download_eurostat]  [download_who]
       └──────────────┴──────────────┴──────────────────┘
                               │
                    [validate_downloads]  ← gate: blocks Spark if any source failed
                               │
         ┌─────────────────────┼──────────────────┐
  [spark_clean_eea]  [spark_clean_era5]  [spark_clean_health]
         └─────────────────────┴──────────────────┘
                               │
                    [spark_join_features]  ← builds feature store
                               │
                    [train_models]         ← only if new data arrived
                               │
                    [notify_complete]      ← optional: log summary

Incremental run logic
---------------------
The pipeline runs weekly. On each run, the download tasks check whether
new data is available (Last-Modified header for EEA/ERA5, age check for
EUROSTAT). If no source downloaded new data, we skip the Spark jobs and
model retraining entirely — the existing feature store and models are
still valid.

This is controlled by the 'new_data_available' flag, which is the return
value of validate_downloads() and is checked before the Spark tasks run.

Scheduling
----------
The flow is scheduled to run every Monday at 06:00 UTC, after EEA typically
publishes the previous week's E2a (unverified) data.
Adjust the cron string or use prefect.yaml for deployment configuration.
"""

import subprocess
import sys
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.schedules import CronSchedule
from datetime import timedelta

from src.ingestion.download_eea       import download_eea
from src.ingestion.download_era5      import download_era5
from src.ingestion.download_eurostat  import download_eurostat
from src.ingestion.download_who       import download_who
from src.ingestion.validate_downloads import validate_downloads
from src.utils.paths import (
    FLAG_EEA_OK, FLAG_ERA5_OK, FLAG_EUROSTAT_OK, FLAG_WHO_OK,
    PROCESSED_FEATURES,
)


# ---------------------------------------------------------------------------
# Helper: run a Spark job as a subprocess
# ---------------------------------------------------------------------------

def _run_spark_job(script_path: str) -> None:
    """
    Submit a PySpark script via spark-submit.

    We run Spark jobs as subprocesses rather than importing them directly
    because Spark initialises its own JVM and resource manager — running
    multiple SparkSessions in the same Python process causes conflicts.
    Each spark-submit call gets its own isolated Spark context.

    Raises subprocess.CalledProcessError if the job exits with a non-zero
    code, which Prefect will catch and mark the task as failed.
    """
    cmd = ["spark-submit", "--master", "local[*]", script_path]
    result = subprocess.run(cmd, check=True, capture_output=False)


# ---------------------------------------------------------------------------
# Ingestion tasks
# ---------------------------------------------------------------------------

@task(
    name="download_eea",
    retries=3,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=12),
    description="Download EEA air quality Parquet files for all five pollutants",
)
def task_download_eea(dry_run: bool = False) -> dict:
    """
    Wraps download_eea() as a Prefect task.

    retries=3: EEA's bulk download endpoint is occasionally unavailable.
    cache: if this task ran successfully in the last 12 hours, skip it —
    EEA publishes new data at most once per day.
    """
    logger = get_run_logger()
    logger.info("Starting EEA download")
    summary = download_eea(dry_run=dry_run)
    logger.info("EEA download summary: %s", summary)
    return summary


@task(
    name="download_era5",
    retries=2,
    retry_delay_seconds=120,
    description="Request ERA5 monthly NetCDF files from Copernicus CDS",
)
def task_download_era5(dry_run: bool = False) -> dict:
    """
    Wraps download_era5() as a Prefect task.

    retries=2 with 120s delay: CDS jobs are queued server-side and can take
    minutes. A retry after 2 minutes catches transient queue failures without
    hammering the CDS API.
    """
    logger = get_run_logger()
    logger.info("Starting ERA5 download")
    summary = download_era5(dry_run=dry_run)
    logger.info("ERA5 download summary: %s", summary)
    return summary


@task(
    name="download_eurostat",
    retries=2,
    retry_delay_seconds=30,
    description="Fetch EUROSTAT health outcome tables (mortality, hospitalisations)",
)
def task_download_eurostat(dry_run: bool = False) -> dict:
    logger = get_run_logger()
    logger.info("Starting EUROSTAT download")
    summary = download_eurostat(dry_run=dry_run)
    logger.info("EUROSTAT download summary: %s", summary)
    return summary


@task(
    name="download_who",
    retries=2,
    retry_delay_seconds=30,
    description="Download WHO European Mortality Database ZIP and extract CSVs",
)
def task_download_who(dry_run: bool = False) -> dict:
    logger = get_run_logger()
    logger.info("Starting WHO download")
    summary = download_who(dry_run=dry_run)
    logger.info("WHO download summary: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Validation gate task
# ---------------------------------------------------------------------------

@task(
    name="validate_downloads",
    retries=1,
    retry_delay_seconds=30,
    description="Schema + size checks on all raw files; writes flag files that gate Spark jobs",
)
def task_validate_downloads(
    eea_summary: dict,
    era5_summary: dict,
    eurostat_summary: dict,
    who_summary: dict,
) -> bool:
    """
    Validates all downloaded files and returns a boolean indicating whether
    new data arrived (True = at least one source downloaded something new).

    This task explicitly depends on all four download tasks via its arguments —
    Prefect will not run this task until all four downloads complete.

    Returns False if:
    - Any validation check fails (corrupted / missing file)
    - No source downloaded new data (incremental: nothing changed)

    Downstream Spark tasks check this return value and skip if False.
    """
    logger = get_run_logger()

    all_passed = validate_downloads()
    if not all_passed:
        logger.error("Validation failed — Spark jobs will be skipped")
        return False

    # Check if any source actually downloaded new data
    new_data = any(
        len(s.get("downloaded", [])) > 0
        for s in [eea_summary, era5_summary, eurostat_summary, who_summary]
    )

    if not new_data:
        logger.info("No new data downloaded — Spark jobs and retraining will be skipped")
    else:
        logger.info("New data available — proceeding to Spark processing")

    return new_data


# ---------------------------------------------------------------------------
# Spark processing tasks
# ---------------------------------------------------------------------------

@task(
    name="spark_clean_eea",
    retries=1,
    retry_delay_seconds=60,
    description="Clean and repartition raw EEA Parquet files",
)
def task_spark_clean_eea(new_data: bool) -> None:
    """
    Only runs if new_data=True (set by validate_downloads).
    Submits spark_clean_eea.py as a spark-submit subprocess.
    """
    if not new_data:
        get_run_logger().info("SKIP: no new data — spark_clean_eea not needed")
        return
    _run_spark_job("src/spark/spark_clean_eea.py")


@task(
    name="spark_clean_era5",
    retries=1,
    retry_delay_seconds=60,
    description="Convert ERA5 NetCDF → Parquet and interpolate to EEA station coordinates",
)
def task_spark_clean_era5(new_data: bool) -> None:
    if not new_data:
        get_run_logger().info("SKIP: no new data — spark_clean_era5 not needed")
        return
    _run_spark_job("src/spark/spark_clean_era5.py")


@task(
    name="spark_clean_health",
    retries=1,
    retry_delay_seconds=60,
    description="Harmonise EUROSTAT and WHO health outcome tables",
)
def task_spark_clean_health(new_data: bool) -> None:
    if not new_data:
        get_run_logger().info("SKIP: no new data — spark_clean_health not needed")
        return
    _run_spark_job("src/spark/spark_clean_health.py")


@task(
    name="spark_join_features",
    retries=1,
    retry_delay_seconds=60,
    description="Join all sources into the feature store; compute lag, rolling, and target features",
)
def task_spark_join_features(
    eea_done,
    era5_done,
    health_done,
    new_data: bool,
) -> None:
    """
    Depends explicitly on all three cleaning tasks (eea_done, era5_done, health_done)
    so Prefect waits for all three to complete before running this join.
    """
    if not new_data:
        get_run_logger().info("SKIP: no new data — spark_join_features not needed")
        return
    _run_spark_job("src/spark/spark_join_features.py")


# ---------------------------------------------------------------------------
# Model training task
# ---------------------------------------------------------------------------

@task(
    name="train_models",
    retries=1,
    retry_delay_seconds=30,
    description="Train LightGBM and XGBoost classifiers; log runs to MLflow",
)
def task_train_models(features_done, new_data: bool) -> None:
    """
    Runs src/model/train.py, which:
      1. Reads the feature store from data/processed/features/
      2. Applies a strict temporal train/validation split
      3. Trains LightGBM (primary) and XGBoost (comparative)
      4. Tunes decision threshold to achieve recall >= 90%
      5. Logs hyperparameters, metrics, and model artefacts to MLflow
      6. Saves the best model to data/models/

    Only runs if new data arrived — no point retraining on identical data.
    """
    if not new_data:
        get_run_logger().info("SKIP: no new data — model retraining not needed")
        return

    result = subprocess.run(
        [sys.executable, "src/model/train.py"],
        check=True,
        capture_output=False,
    )


# ---------------------------------------------------------------------------
# Notification task (optional — logs a run summary)
# ---------------------------------------------------------------------------

@task(name="notify_complete", description="Log pipeline completion summary")
def task_notify_complete(new_data: bool, training_done=None) -> None:
    logger = get_run_logger()
    if new_data:
        logger.info("Pipeline run COMPLETE — data updated, models retrained")
    else:
        logger.info("Pipeline run COMPLETE — no new data, existing artefacts unchanged")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

@flow(
    name="air_quality_pipeline",
    description=(
        "Weekly pipeline: download EEA + ERA5 + EUROSTAT + WHO data, "
        "clean with Spark, build feature store, retrain PM2.5 exceedance models"
    ),
)
def air_quality_pipeline(dry_run: bool = False) -> None:
    """
    Full pipeline flow — runs all stages end to end.

    Parameters
    ----------
    dry_run : if True, all download tasks run in dry-run mode (no files written).
              Useful for testing the flow structure without downloading data.

    To run manually:
        from src.pipeline.flow import air_quality_pipeline
        air_quality_pipeline()

    To run with dry-run:
        air_quality_pipeline(dry_run=True)
    """
    # Stage 1: Download all sources in parallel
    # Prefect runs these concurrently because they have no dependencies on each other
    eea_summary       = task_download_eea(dry_run=dry_run)
    era5_summary      = task_download_era5(dry_run=dry_run)
    eurostat_summary  = task_download_eurostat(dry_run=dry_run)
    who_summary       = task_download_who(dry_run=dry_run)

    # Stage 2: Validate — gate that blocks everything downstream
    new_data = task_validate_downloads(
        eea_summary, era5_summary, eurostat_summary, who_summary
    )

    # Stage 3: Spark cleaning — runs in parallel, each independent
    eea_done    = task_spark_clean_eea(new_data)
    era5_done   = task_spark_clean_era5(new_data)
    health_done = task_spark_clean_health(new_data)

    # Stage 4: Feature store join — waits for all three cleaning jobs
    features_done = task_spark_join_features(eea_done, era5_done, health_done, new_data)

    # Stage 5: Model retraining
    training_done = task_train_models(features_done, new_data)

    # Stage 6: Completion notification
    task_notify_complete(new_data, training_done)


# ---------------------------------------------------------------------------
# Deployment entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run immediately (useful for testing)
    air_quality_pipeline()
