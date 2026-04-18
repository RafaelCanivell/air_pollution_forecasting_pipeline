"""
download_era5.py
----------------
Downloads ERA5 reanalysis data (meteorological variables) from the
Copernicus Climate Data Store (CDS) via the official `cdsapi` client.

Data source
-----------
Copernicus/ECMWF — ERA5 hourly data on single levels
https://cds.climate.copernicus.eu

Variables downloaded
--------------------
We download the six meteorological variables used as features in the
PM2.5 prediction model and as the causal instrument (boundary layer height):

  - 2m_temperature            → daily temperature (drives thermal inversions)
  - 10m_u_component_of_wind   → east-west wind (pollutant dispersion)
  - 10m_v_component_of_wind   → north-south wind (pollutant dispersion)
  - surface_pressure          → atmospheric pressure
  - total_precipitation       → precipitation (washes out particles)
  - boundary_layer_height     → KEY causal instrument for the IV analysis

File format and structure
-------------------------
CDS returns NetCDF files.  We request one file per (year, month) to keep
individual file sizes manageable and to support incremental downloads.

Output: data/raw/era5/era5_<YYYY>_<MM>.nc

Design decisions
----------------
- We request hourly data and aggregate to daily means in the Spark job
  (spark_clean_era5.py), not here.  Keeping raw = raw is a core MLOps
  principle: never transform during ingestion.
- We limit the bounding box to Europe (lon -25→45, lat 34→72) to avoid
  downloading global data we won't use.
- Year/month combinations already on disk are skipped (incremental runs).
- CDS jobs are queued server-side; the client polls until ready.
  The @retry decorator handles transient network failures during polling.

Setup required (one-time)
--------------------------
    pip install cdsapi
    # Create ~/.cdsapirc with:
    # url: https://cds.climate.copernicus.eu/api/v2
    # key: <your_uid>:<your_api_key>
"""

import argparse
import os
from pathlib import Path

import cdsapi

from src.utils.logging_config import get_logger
from src.utils.paths import RAW_ERA5
from src.utils.retry import retry

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "total_precipitation",
    "boundary_layer_height",   # causal instrument for IV analysis
]

# European bounding box: [north, west, south, east]
AREA_EUROPE = [72, -25, 34, 45]

# Download range — adjust END_YEAR to extend the dataset
START_YEAR = 2019
END_YEAR   = 2024

# CDS dataset identifier
CDS_DATASET = "reanalysis-era5-single-levels"


# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------

@retry(max_attempts=3, base_delay=30.0, exceptions=(Exception,))
def _download_month(
    client: cdsapi.Client,
    year: int,
    month: int,
    dest: Path,
    dry_run: bool = False,
) -> bool:
    """
    Request one (year, month) NetCDF file from the CDS API.

    Returns True if downloaded, False if skipped.

    Note on CDS behaviour: the client.retrieve() call submits a job to the
    CDS queue and blocks until the file is ready.  Large requests (full year,
    many variables) can queue for 10–30 minutes.  Requesting month by month
    keeps each job small and avoids timeouts.
    """
    if dest.exists():
        log.info("SKIP (exists): %s", dest.name)
        return False

    if dry_run:
        log.info("DRY-RUN — would request: %s", dest.name)
        return False

    log.info("Requesting CDS: %s", dest.name)

    # Build list of all days in this month as zero-padded strings
    import calendar
    days_in_month = calendar.monthrange(year, month)[1]
    days = [f"{d:02d}" for d in range(1, days_in_month + 1)]

    client.retrieve(
        CDS_DATASET,
        {
            "product_type": "reanalysis",
            "variable": VARIABLES,
            "year":  str(year),
            "month": f"{month:02d}",
            "day":   days,
            "time":  [f"{h:02d}:00" for h in range(24)],  # all 24 hours
            "area":  AREA_EUROPE,
            "format": "netcdf",
        },
        str(dest),  # cdsapi writes directly to this path
    )

    size_mb = dest.stat().st_size / 1024 / 1024
    log.info("Saved: %s  (%.1f MB)", dest.name, size_mb)
    return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def download_era5(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    dry_run: bool = False,
) -> dict:
    """
    Download all ERA5 monthly files for the configured year range.

    Returns a summary dict {"downloaded": [...], "skipped": [...], "failed": [...]}.

    Initialises the cdsapi.Client once and reuses it across all requests —
    each instantiation reads ~/.cdsapirc, so one init is enough.
    """
    summary = {"downloaded": [], "skipped": [], "failed": []}

    # Initialise CDS client — reads credentials from ~/.cdsapirc
    # quiet=True suppresses the verbose CDS banner on every call
    client = cdsapi.Client(quiet=True)

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            dest = RAW_ERA5 / f"era5_{year}_{month:02d}.nc"
            try:
                downloaded = _download_month(client, year, month, dest, dry_run)
                key = "downloaded" if downloaded else "skipped"
                summary[key].append(dest.name)
            except Exception as exc:
                log.error("FAILED: %s — %s", dest.name, exc)
                summary["failed"].append(dest.name)

    log.info(
        "ERA5 download complete — downloaded: %d | skipped: %d | failed: %d",
        len(summary["downloaded"]),
        len(summary["skipped"]),
        len(summary["failed"]),
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5 meteorological data")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year",   type=int, default=END_YEAR)
    args = parser.parse_args()

    download_era5(
        start_year=args.start_year,
        end_year=args.end_year,
        dry_run=args.dry_run,
    )
