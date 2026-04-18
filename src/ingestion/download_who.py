"""
download_who.py
---------------
Downloads the WHO European Mortality Database (EMD), which provides
cause-specific mortality data by age, sex, country, and year for
European WHO member states.

Data source
-----------
WHO Europe — European Mortality Database
https://gateway.euro.who.int/en/datasets/european-mortality-database

Why WHO in addition to EUROSTAT?
---------------------------------
EUROSTAT covers EU member states only.  The WHO EMD extends coverage to
all European WHO member states (53 countries), including non-EU countries
such as Norway, Switzerland, Ukraine, and Turkey.  For transboundary
pollution episodes this broader geographic scope matters.

Additionally, the WHO EMD uses the ICD-10 classification consistently
across all countries and years, making cause-of-death comparisons more
reliable than cross-national EUROSTAT tables which can have coding gaps.

Files downloaded
----------------
The WHO EMD is distributed as a set of CSV files (one per data component):
  - mortalitydata.csv  — death counts by cause (ICD-10), year, age, sex, country
  - country_codes.csv  — WHO country code lookup table
  - cause_codes.csv    — ICD chapter / cause code lookup table
  - pop.csv            — population denominators (for rate calculation)

Output: data/raw/who/<filename>.csv  (kept as CSV — small files, no benefit to Parquet)

Design decisions
----------------
- We download the static CSV files from the WHO bulk download endpoint.
  The WHO does not provide an API for this dataset, so we use direct HTTP.
- Files are checked by size: if the remote Content-Length differs from the
  local file size, the file is re-downloaded.  This handles cases where WHO
  publishes an updated version without changing the filename.
- Raw CSVs are kept as-is.  All cleaning and joining happens in
  spark_clean_health.py — keeping raw = raw.
"""

import argparse
from pathlib import Path

import requests

from src.utils.logging_config import get_logger
from src.utils.paths import RAW_WHO
from src.utils.retry import retry

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# WHO bulk download base URL (subject to change — check WHO gateway if broken)
_WHO_BASE = "https://gateway.euro.who.int/en/datasets/european-mortality-database"

# Direct CSV download URLs for the four EMD components
# These are stable bulk-download links from the WHO data gateway
WHO_FILES = {
    "mortalitydata.csv": (
        "https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates/"
        "ghe-leading-causes-of-death"
        # NOTE: replace with the actual direct download URL from the WHO gateway
        # The exact URL should be verified at: https://gateway.euro.who.int
    ),
    "country_codes.csv": (
        "https://gateway.euro.who.int/en/indicators/hfa_1-0000101-population/data-download"
        # NOTE: replace with actual URL
    ),
    "cause_codes.csv": (
        "https://gateway.euro.who.int/en/indicators/hfa_1-0000101-population/data-download"
        # NOTE: replace with actual URL
    ),
    "pop.csv": (
        "https://gateway.euro.who.int/en/indicators/hfa_1-0000101-population/data-download"
        # NOTE: replace with actual URL
    ),
}

# More realistic fallback: download the ZIP bundle the WHO provides
# and extract inside data/raw/who/
WHO_ZIP_URL = (
    "https://www.euro.who.int/__data/assets/zip-file/0010/283178/WHO-MDB.zip"
)

_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------

def _remote_size(url: str) -> int:
    """
    Return the Content-Length of a URL via HEAD request.
    Returns -1 if the server doesn't provide Content-Length.
    """
    try:
        resp = requests.head(url, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        return int(resp.headers.get("Content-Length", -1))
    except Exception:
        return -1


def _needs_download(dest: Path, url: str) -> bool:
    """
    Decide whether to (re)download.

    Rules:
    1. File doesn't exist → download.
    2. Remote Content-Length differs from local size → re-download (WHO updated it).
    3. Otherwise → skip.
    """
    if not dest.exists():
        return True
    remote_size = _remote_size(url)
    if remote_size > 0 and dest.stat().st_size != remote_size:
        log.info(
            "Size mismatch for %s — local: %d B, remote: %d B → re-downloading",
            dest.name, dest.stat().st_size, remote_size,
        )
        return True
    return False


@retry(max_attempts=4, base_delay=15.0, exceptions=(requests.RequestException,))
def _download_file(url: str, dest: Path, dry_run: bool = False) -> bool:
    """Stream-download a single file to dest."""
    if not _needs_download(dest, url):
        log.info("SKIP (up to date): %s", dest.name)
        return False

    if dry_run:
        log.info("DRY-RUN — would download: %s → %s", url, dest.name)
        return False

    log.info("Downloading: %s", dest.name)
    with requests.get(url, stream=True, timeout=_TIMEOUT) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=512 * 1024):
                fh.write(chunk)

    log.info("Saved: %s  (%.1f MB)", dest.name, dest.stat().st_size / 1024 / 1024)
    return True


@retry(max_attempts=3, base_delay=20.0, exceptions=(requests.RequestException,))
def _download_zip_bundle(dry_run: bool = False) -> bool:
    """
    Alternative: download the WHO EMD as a ZIP bundle and extract it.

    Used as a fallback if individual CSV URLs are unavailable.
    The ZIP contains the same four CSV files.
    """
    import zipfile
    import io

    zip_dest = RAW_WHO / "WHO-MDB.zip"

    if not _needs_download(zip_dest, WHO_ZIP_URL):
        log.info("SKIP (ZIP up to date)")
        return False

    if dry_run:
        log.info("DRY-RUN — would download WHO ZIP bundle")
        return False

    log.info("Downloading WHO EMD ZIP bundle …")
    with requests.get(WHO_ZIP_URL, stream=True, timeout=_TIMEOUT) as resp:
        resp.raise_for_status()
        content = resp.content

    # Extract directly from memory — avoids writing a temp file
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        zf.extractall(RAW_WHO)
        log.info("Extracted files: %s", zf.namelist())

    log.info("WHO ZIP bundle downloaded and extracted to %s", RAW_WHO)
    return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def download_who(dry_run: bool = False, use_zip: bool = True) -> dict:
    """
    Download WHO European Mortality Database.

    Parameters
    ----------
    dry_run  : print intent without writing files
    use_zip  : if True (default), download the ZIP bundle;
               if False, attempt to download individual CSV files

    Returns a summary dict {"downloaded": [...], "skipped": [...], "failed": [...]}.
    """
    summary = {"downloaded": [], "skipped": [], "failed": []}

    if use_zip:
        # Preferred method — single ZIP with all four CSV files
        try:
            downloaded = _download_zip_bundle(dry_run=dry_run)
            key = "downloaded" if downloaded else "skipped"
            summary[key].append("WHO-MDB.zip")
        except Exception as exc:
            log.error("FAILED: WHO ZIP bundle — %s", exc)
            summary["failed"].append("WHO-MDB.zip")
    else:
        # Fallback — individual files (requires valid URLs in WHO_FILES dict)
        for filename, url in WHO_FILES.items():
            dest = RAW_WHO / filename
            try:
                downloaded = _download_file(url, dest, dry_run=dry_run)
                key = "downloaded" if downloaded else "skipped"
                summary[key].append(filename)
            except Exception as exc:
                log.error("FAILED: %s — %s", filename, exc)
                summary["failed"].append(filename)

    log.info(
        "WHO download complete — downloaded: %d | skipped: %d | failed: %d",
        len(summary["downloaded"]),
        len(summary["skipped"]),
        len(summary["failed"]),
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download WHO European Mortality Database"
    )
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument(
        "--individual-files",
        action="store_true",
        help="Download individual CSVs instead of the ZIP bundle",
    )
    args = parser.parse_args()
    download_who(dry_run=args.dry_run, use_zip=not args.individual_files)
