"""
download_eea.py
---------------
Downloads EEA Air Quality e-Reporting bulk data for five pollutants
(PM2.5, PM10, NO2, O3, SO2) and saves them as Parquet files under data/raw/eea/.

Data source
-----------
EEA Air Quality Download Service
https://www.eea.europa.eu/en/datahub/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d

The service returns pre-built Parquet files partitioned by pollutant and
verification status (E1a = verified, E2a = up-to-date unverified).
We download both verification levels so the pipeline can use the most
recent unverified data while also keeping the stable verified archive.

Design decisions
----------------
- Each pollutant is downloaded independently so a single failure doesn't
  abort the whole run.
- Files are only re-downloaded if the remote file is newer than the local
  copy (Last-Modified header check) — supports incremental runs.
- The --dry-run flag prints what would be downloaded without writing to disk,
  matching the Quick Start instructions in the README.
- Retry logic is handled by the @retry decorator from utils/retry.py.

Output
------
data/raw/eea/<pollutant>_<verification>.parquet
e.g. data/raw/eea/PM2.5_E2a.parquet
"""

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.utils.logging_config import get_logger
from src.utils.paths import RAW_EEA
from src.utils.retry import retry

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Base URL for the EEA bulk download service.
# Each pollutant has two files: E1a (verified) and E2a (unverified/recent).
_BASE_URL = (
    "https://eeadmz1-downloads-webapp.azurewebsites.net/download-data?"
    "pollutant={pollutant}&reportingPeriod=2013-2024&"
    "aggregationType=day&fileFormat=parquet&verification={verification}"
)

POLLUTANTS    = ["PM2.5", "PM10", "NO2", "O3", "SO2"]
VERIFICATIONS = ["E1a", "E2a"]   # E1a = verified historic, E2a = recent unverified

# Request timeout in seconds — EEA files can be large
_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------

@retry(max_attempts=4, base_delay=10.0, exceptions=(requests.RequestException,))
def _download_one(url: str, dest: Path, dry_run: bool = False) -> bool:
    """
    Download a single file from `url` to `dest`.

    Returns True if the file was (re)downloaded, False if it was skipped
    because the local copy is already up to date.

    The Last-Modified header is used to avoid re-downloading unchanged files.
    This is the key mechanism that makes incremental pipeline runs cheap.
    """
    # --- Check if we already have an up-to-date local copy ------------------
    if dest.exists():
        # Send a HEAD request first — no body, just headers
        head = requests.head(url, timeout=_TIMEOUT)
        head.raise_for_status()

        remote_modified_str = head.headers.get("Last-Modified")
        if remote_modified_str:
            remote_dt = datetime.strptime(
                remote_modified_str, "%a, %d %b %Y %H:%M:%S %Z"
            ).replace(tzinfo=timezone.utc)
            local_dt = datetime.fromtimestamp(
                dest.stat().st_mtime, tz=timezone.utc
            )
            if local_dt >= remote_dt:
                log.info("SKIP (up to date): %s", dest.name)
                return False

    if dry_run:
        log.info("DRY-RUN — would download: %s → %s", url, dest)
        return False

    # --- Stream download — avoids loading the entire file into memory -------
    log.info("Downloading → %s", dest.name)
    with requests.get(url, stream=True, timeout=_TIMEOUT) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0

        with open(dest, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                fh.write(chunk)
                downloaded += len(chunk)

        log.info(
            "Saved: %s  (%.1f MB)",
            dest.name,
            downloaded / 1024 / 1024,
        )
    return True


# ---------------------------------------------------------------------------
# Public entry point — called directly and by pipeline/flow.py
# ---------------------------------------------------------------------------

def download_eea(dry_run: bool = False) -> dict:
    """
    Download all EEA pollutant files.

    Returns a summary dict:
        {"downloaded": [...], "skipped": [...], "failed": [...]}

    Returning a structured result (rather than just logging) lets the
    Prefect task in flow.py inspect outcomes and decide whether to trigger
    a Spark re-run.
    """
    summary = {"downloaded": [], "skipped": [], "failed": []}

    for pollutant in POLLUTANTS:
        for verification in VERIFICATIONS:
            url  = _BASE_URL.format(pollutant=pollutant, verification=verification)
            dest = RAW_EEA / f"{pollutant}_{verification}.parquet"

            try:
                downloaded = _download_one(url, dest, dry_run=dry_run)
                key = "downloaded" if downloaded else "skipped"
                summary[key].append(dest.name)
            except Exception as exc:
                log.error("FAILED: %s — %s", dest.name, exc)
                summary["failed"].append(dest.name)

    # Log a clean summary at the end
    log.info(
        "EEA download complete — downloaded: %d | skipped: %d | failed: %d",
        len(summary["downloaded"]),
        len(summary["skipped"]),
        len(summary["failed"]),
    )
    if summary["failed"]:
        log.warning("Failed files: %s", summary["failed"])

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download EEA air quality data")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without writing any files",
    )
    args = parser.parse_args()
    download_eea(dry_run=args.dry_run)
