"""
download_eurostat.py
--------------------
Downloads three EUROSTAT health datasets used as outcome variables in the
causal inference analyses.

Data source
-----------
EUROSTAT — European Statistics
https://ec.europa.eu/eurostat

Datasets downloaded
-------------------
1. demo_r_mweek3   — Weekly deaths by NUTS3 region (all causes)
   Used in DiD Analysis 1: PM2.5 episodes → respiratory mortality.
   The fine geographic resolution (NUTS3) is essential for matching treated
   and control regions at sub-national level.

2. hlth_cd_aro     — Annual deaths by cause, age, sex, and NUTS2 region
   Used to isolate respiratory and cardiovascular deaths specifically.
   Coarser geography (NUTS2) but richer cause-of-death detail.

3. hlth_co_hospit  — Hospital admissions by diagnosis, age, and country
   Used in IV Analysis 2: PM2.5 → cardiovascular hospitalisations.

Why `eurostat` package
----------------------
The `eurostat` Python package wraps the EUROSTAT JSON-stat API, handles
pagination, and returns a clean pandas DataFrame.  This avoids manual URL
construction and parsing of the API's complex JSON format.

Output
------
Each dataset is saved as a Parquet file (more efficient than CSV for wide
EUROSTAT tables, which can have thousands of columns):
  data/raw/eurostat/demo_r_mweek3/demo_r_mweek3.parquet
  data/raw/eurostat/hlth_cd_aro/hlth_cd_aro.parquet
  data/raw/eurostat/hlth_co_hospit/hlth_co_hospit.parquet

Design decisions
----------------
- We do not filter by country or year at download time — we keep the full
  table and let the Spark job apply filters.  This preserves flexibility:
  if we want to add a new country to the analysis, we don't need to re-download.
- EUROSTAT tables are updated infrequently (quarterly at most), so we check
  the file's age and skip if it is less than 30 days old.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import eurostat          # pip install eurostat
import pandas as pd

from src.utils.logging_config import get_logger
from src.utils.paths import RAW_EUROSTAT_MORT, RAW_EUROSTAT_CAUSE, RAW_EUROSTAT_HOSP
from src.utils.retry import retry

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Each entry: (dataset_code, destination_directory, human_readable_description)
DATASETS = [
    (
        "demo_r_mweek3",
        RAW_EUROSTAT_MORT,
        "Weekly deaths by NUTS3 region",
    ),
    (
        "hlth_cd_aro",
        RAW_EUROSTAT_CAUSE,
        "Deaths by respiratory/cardiovascular cause",
    ),
    (
        "hlth_co_hospit",
        RAW_EUROSTAT_HOSP,
        "Hospital admissions by diagnosis",
    ),
]

# Skip re-download if local file is younger than this many days
_CACHE_DAYS = 30


# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------

def _is_stale(path: Path, max_age_days: int = _CACHE_DAYS) -> bool:
    """Return True if the file does not exist or is older than max_age_days."""
    if not path.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age > timedelta(days=max_age_days)


@retry(max_attempts=3, base_delay=15.0, exceptions=(Exception,))
def _download_one(
    code: str,
    dest_dir: Path,
    description: str,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    """
    Fetch one EUROSTAT dataset and save it as Parquet.

    Parameters
    ----------
    code        : EUROSTAT dataset code (e.g. "demo_r_mweek3")
    dest_dir    : directory where the Parquet file will be saved
    description : human-readable label for log messages
    dry_run     : if True, print intent without downloading
    force       : if True, re-download even if local copy is recent

    Returns True if downloaded, False if skipped.
    """
    dest = dest_dir / f"{code}.parquet"

    if not force and not _is_stale(dest):
        log.info("SKIP (fresh, < %d days old): %s", _CACHE_DAYS, dest.name)
        return False

    if dry_run:
        log.info("DRY-RUN — would download: %s (%s)", code, description)
        return False

    log.info("Downloading EUROSTAT: %s — %s", code, description)

    # get_data_df returns a pandas DataFrame with a MultiIndex
    # (dimensions as index levels, years/periods as columns)
    df = eurostat.get_data_df(code)

    if df is None or df.empty:
        raise ValueError(f"EUROSTAT returned empty DataFrame for {code}")

    # Reset index so all dimension columns become regular columns —
    # cleaner for Spark to read and avoids MultiIndex serialisation issues
    df = df.reset_index()

    df.to_parquet(dest, index=False, engine="pyarrow")

    size_mb = dest.stat().st_size / 1024 / 1024
    log.info(
        "Saved: %s  (%.1f MB, %d rows x %d cols)",
        dest.name, size_mb, len(df), len(df.columns),
    )
    return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def download_eurostat(dry_run: bool = False, force: bool = False) -> dict:
    """
    Download all three EUROSTAT datasets.

    Returns a summary dict {"downloaded": [...], "skipped": [...], "failed": [...]}.
    """
    summary = {"downloaded": [], "skipped": [], "failed": []}

    for code, dest_dir, description in DATASETS:
        try:
            downloaded = _download_one(
                code, dest_dir, description,
                dry_run=dry_run, force=force,
            )
            key = "downloaded" if downloaded else "skipped"
            summary[key].append(code)
        except Exception as exc:
            log.error("FAILED: %s — %s", code, exc)
            summary["failed"].append(code)

    log.info(
        "EUROSTAT download complete — downloaded: %d | skipped: %d | failed: %d",
        len(summary["downloaded"]),
        len(summary["skipped"]),
        len(summary["failed"]),
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download EUROSTAT health datasets")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if local copy is recent",
    )
    args = parser.parse_args()
    download_eurostat(dry_run=args.dry_run, force=args.force)
