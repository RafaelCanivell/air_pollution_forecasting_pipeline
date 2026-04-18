"""
paths.py
--------
Single source of truth for all filesystem paths in the project.

Why this exists
---------------
Hard-coding paths in individual scripts creates maintenance debt: if the
project root moves, or if you switch from a local run to a Docker container,
you would need to hunt down every script.  Importing from here means you
change one file and everything stays consistent.

Usage
-----
    from src.utils.paths import RAW_EEA, PROCESSED_FEATURES
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root  (two levels above this file: src/utils/paths.py → project/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Raw data  — exactly mirrors the structure documented in the README
# ---------------------------------------------------------------------------
DATA_RAW          = PROJECT_ROOT / "data" / "raw"

RAW_EEA           = DATA_RAW / "eea"
RAW_ERA5          = DATA_RAW / "era5"
RAW_EUROSTAT      = DATA_RAW / "eurostat"
RAW_EUROSTAT_MORT = RAW_EUROSTAT / "demo_r_mweek3"   # weekly mortality by NUTS3
RAW_EUROSTAT_CAUSE= RAW_EUROSTAT / "hlth_cd_aro"      # deaths by respiratory/cardiovascular cause
RAW_EUROSTAT_HOSP = RAW_EUROSTAT / "hlth_co_hospit"   # hospital admissions
RAW_WHO           = DATA_RAW / "who"

# ---------------------------------------------------------------------------
# Processed data
# ---------------------------------------------------------------------------
DATA_PROCESSED       = PROJECT_ROOT / "data" / "processed"

PROCESSED_EEA        = DATA_PROCESSED / "eea"
PROCESSED_ERA5       = DATA_PROCESSED / "era5"
PROCESSED_HEALTH     = DATA_PROCESSED / "health"
PROCESSED_FEATURES   = DATA_PROCESSED / "features"
PROCESSED_AGGREGATIONS = DATA_PROCESSED / "aggregations"

# ---------------------------------------------------------------------------
# Models and logs
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR   = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Validation flag files
# Written by validate_downloads.py; read by pipeline/flow.py before Spark jobs
# ---------------------------------------------------------------------------
FLAG_EEA_OK       = DATA_RAW / ".eea_validated"
FLAG_ERA5_OK      = DATA_RAW / ".era5_validated"
FLAG_EUROSTAT_OK  = DATA_RAW / ".eurostat_validated"
FLAG_WHO_OK       = DATA_RAW / ".who_validated"

# ---------------------------------------------------------------------------
# Helper: create all directories on import so scripts never crash on mkdir
# ---------------------------------------------------------------------------
_ALL_DIRS = [
    RAW_EEA, RAW_ERA5, RAW_EUROSTAT,
    RAW_EUROSTAT_MORT, RAW_EUROSTAT_CAUSE, RAW_EUROSTAT_HOSP,
    RAW_WHO,
    PROCESSED_EEA, PROCESSED_ERA5, PROCESSED_HEALTH,
    PROCESSED_FEATURES, PROCESSED_AGGREGATIONS,
    MODELS_DIR, LOGS_DIR,
]

for _d in _ALL_DIRS:
    _d.mkdir(parents=True, exist_ok=True)
