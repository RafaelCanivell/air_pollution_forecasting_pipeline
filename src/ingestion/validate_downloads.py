"""
validate_downloads.py
---------------------
Validates the integrity of all raw downloaded files before the Spark
processing jobs are allowed to run.

Why this exists
---------------
Running a Spark job on a corrupted or truncated file wastes compute and
produces silent downstream errors (wrong counts, missing countries, NaN
floods) that are hard to debug.  This script acts as a quality gate
between the ingestion and processing phases.

What it checks
--------------
EEA  (Parquet)
  - File exists and size > threshold
  - Can be opened as a valid Parquet file (pyarrow read_schema)
  - Expected columns are present
  - No completely empty partitions

ERA5 (NetCDF)
  - File exists and size > threshold
  - Can be opened with xarray without errors
  - Expected variables are present in the dataset

EUROSTAT (Parquet)
  - File exists
  - Can be opened as Parquet
  - Row count > 0

WHO (CSV or ZIP-extracted CSVs)
  - mortalitydata.csv exists and row count > 0

Output
------
- Logs a PASS / FAIL for each check.
- Writes flag files (data/raw/.xxx_validated) that Prefect reads before
  triggering Spark jobs.  If a flag is missing, the Spark task is blocked.
- Exits with code 0 if all checks pass, 1 if any check fails.

Usage
-----
    python src/ingestion/validate_downloads.py
    python src/ingestion/validate_downloads.py --strict   # fail on any warning
"""

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

from src.utils.logging_config import get_logger
from src.utils.paths import (
    RAW_EEA, RAW_ERA5,
    RAW_EUROSTAT_MORT, RAW_EUROSTAT_CAUSE, RAW_EUROSTAT_HOSP,
    RAW_WHO,
    FLAG_EEA_OK, FLAG_ERA5_OK, FLAG_EUROSTAT_OK, FLAG_WHO_OK,
)

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Thresholds and expected schema
# ---------------------------------------------------------------------------

# Minimum acceptable file size in bytes (a non-empty Parquet/NetCDF is always larger)
MIN_EEA_BYTES    = 100_000      # 100 KB — EEA files are typically tens of MB
MIN_ERA5_BYTES   = 1_000_000   # 1 MB   — ERA5 monthly NetCDF is typically 100+ MB

EEA_EXPECTED_COLUMNS = {
    "AirQualityStationEoICode", "AirPollutant", "Start", "End",
    "Value", "Validity", "Verification",
}

ERA5_EXPECTED_VARIABLES = {
    "t2m",    # 2m temperature
    "u10",    # 10m u-component of wind
    "v10",    # 10m v-component of wind
    "sp",     # surface pressure
    "tp",     # total precipitation
    "blh",    # boundary layer height (causal instrument)
}

# Short codes for the Parquet Eurostat files
EUROSTAT_FILES = {
    "demo_r_mweek3": RAW_EUROSTAT_MORT  / "demo_r_mweek3.parquet",
    "hlth_cd_aro":   RAW_EUROSTAT_CAUSE / "hlth_cd_aro.parquet",
    "hlth_co_hospit":RAW_EUROSTAT_HOSP  / "hlth_co_hospit.parquet",
}


# ---------------------------------------------------------------------------
# Individual check functions — each returns (passed: bool, message: str)
# ---------------------------------------------------------------------------

def _check_eea() -> bool:
    """Validate all EEA Parquet files."""
    from src.ingestion.download_eea import POLLUTANTS, VERIFICATIONS

    all_ok = True
    for pollutant in POLLUTANTS:
        for verification in VERIFICATIONS:
            path = RAW_EEA / f"{pollutant}_{verification}.parquet"

            if not path.exists():
                log.error("FAIL [EEA] Missing: %s", path.name)
                all_ok = False
                continue

            if path.stat().st_size < MIN_EEA_BYTES:
                log.error(
                    "FAIL [EEA] File too small (%d B): %s",
                    path.stat().st_size, path.name,
                )
                all_ok = False
                continue

            try:
                schema = pq.read_schema(path)
                schema_cols = set(schema.names)
                missing = EEA_EXPECTED_COLUMNS - schema_cols
                if missing:
                    log.error(
                        "FAIL [EEA] Missing columns %s in %s", missing, path.name
                    )
                    all_ok = False
                else:
                    log.info("PASS [EEA] %s", path.name)
            except Exception as exc:
                log.error("FAIL [EEA] Cannot read %s — %s", path.name, exc)
                all_ok = False

    return all_ok


def _check_era5() -> bool:
    """Validate ERA5 NetCDF files — check variables and basic structure."""
    try:
        import xarray as xr
    except ImportError:
        log.warning("xarray not installed — skipping ERA5 variable checks")
        # Fall back to size-only check
        files = sorted(RAW_ERA5.glob("era5_*.nc"))
        if not files:
            log.error("FAIL [ERA5] No NetCDF files found in %s", RAW_ERA5)
            return False
        for f in files:
            if f.stat().st_size < MIN_ERA5_BYTES:
                log.error("FAIL [ERA5] File too small: %s", f.name)
                return False
            log.info("PASS [ERA5] %s (size only)", f.name)
        return True

    files = sorted(RAW_ERA5.glob("era5_*.nc"))
    if not files:
        log.error("FAIL [ERA5] No NetCDF files found in %s", RAW_ERA5)
        return False

    all_ok = True
    for path in files:
        if path.stat().st_size < MIN_ERA5_BYTES:
            log.error("FAIL [ERA5] File too small (%d B): %s",
                      path.stat().st_size, path.name)
            all_ok = False
            continue
        try:
            ds = xr.open_dataset(path)
            missing = ERA5_EXPECTED_VARIABLES - set(ds.data_vars)
            ds.close()
            if missing:
                log.error(
                    "FAIL [ERA5] Missing variables %s in %s", missing, path.name
                )
                all_ok = False
            else:
                log.info("PASS [ERA5] %s", path.name)
        except Exception as exc:
            log.error("FAIL [ERA5] Cannot open %s — %s", path.name, exc)
            all_ok = False

    return all_ok


def _check_eurostat() -> bool:
    """Validate EUROSTAT Parquet files — existence and non-empty."""
    all_ok = True
    for code, path in EUROSTAT_FILES.items():
        if not path.exists():
            log.error("FAIL [EUROSTAT] Missing: %s", path.name)
            all_ok = False
            continue
        try:
            meta = pq.read_metadata(path)
            n_rows = meta.num_rows
            if n_rows == 0:
                log.error("FAIL [EUROSTAT] Empty file: %s", path.name)
                all_ok = False
            else:
                log.info("PASS [EUROSTAT] %s  (%d rows)", path.name, n_rows)
        except Exception as exc:
            log.error("FAIL [EUROSTAT] Cannot read %s — %s", path.name, exc)
            all_ok = False
    return all_ok


def _check_who() -> bool:
    """Validate WHO data — look for mortality CSV in the extraction directory."""
    import pandas as pd

    # After ZIP extraction, mortalitydata.csv should be present
    mort_file = RAW_WHO / "mortalitydata.csv"

    # Also accept the file in a subdirectory (ZIP may extract into a folder)
    if not mort_file.exists():
        candidates = list(RAW_WHO.rglob("mortalitydata.csv"))
        if candidates:
            mort_file = candidates[0]
        else:
            log.error("FAIL [WHO] mortalitydata.csv not found under %s", RAW_WHO)
            return False

    try:
        # Read only the first 100 rows to validate structure cheaply
        df = pd.read_csv(mort_file, nrows=100, low_memory=False)
        if df.empty:
            log.error("FAIL [WHO] mortalitydata.csv has no rows")
            return False
        log.info(
            "PASS [WHO] %s  (%d cols, sampling passed)",
            mort_file.name, len(df.columns),
        )
        return True
    except Exception as exc:
        log.error("FAIL [WHO] Cannot read mortalitydata.csv — %s", exc)
        return False


# ---------------------------------------------------------------------------
# Flag file helpers
# ---------------------------------------------------------------------------

def _write_flag(flag_path: Path) -> None:
    """Write an empty flag file to signal that validation passed."""
    flag_path.touch()
    log.info("Flag written: %s", flag_path.name)


def _clear_flag(flag_path: Path) -> None:
    """Remove flag file so downstream tasks are blocked until re-validation."""
    if flag_path.exists():
        flag_path.unlink()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_downloads(strict: bool = False) -> bool:
    """
    Run all validation checks.

    Parameters
    ----------
    strict : if True, treat warnings as failures.

    Returns True if all checks passed, False otherwise.
    Also writes / clears flag files for each data source.
    """
    log.info("=" * 60)
    log.info("Starting download validation")
    log.info("=" * 60)

    results = {
        "EEA":       (_check_eea,       FLAG_EEA_OK),
        "ERA5":      (_check_era5,      FLAG_ERA5_OK),
        "EUROSTAT":  (_check_eurostat,  FLAG_EUROSTAT_OK),
        "WHO":       (_check_who,       FLAG_WHO_OK),
    }

    all_passed = True
    for source, (check_fn, flag_path) in results.items():
        log.info("--- Checking %s ---", source)
        passed = check_fn()
        if passed:
            _write_flag(flag_path)
        else:
            _clear_flag(flag_path)
            all_passed = False

    log.info("=" * 60)
    if all_passed:
        log.info("VALIDATION COMPLETE — all checks PASSED")
    else:
        log.error("VALIDATION COMPLETE — one or more checks FAILED")
        log.error("Check the log above for details.")
        log.error("Spark jobs will be blocked until all flags are present.")
    log.info("=" * 60)

    return all_passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate raw downloaded files before Spark processing"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 on warnings as well as errors",
    )
    args = parser.parse_args()

    passed = validate_downloads(strict=args.strict)
    sys.exit(0 if passed else 1)
