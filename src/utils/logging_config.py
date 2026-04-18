"""
logging_config.py
-----------------
Configures a single logger that every module in the project imports.

Why this exists
---------------
Python's logging module, if left unconfigured, silently swallows messages or
produces duplicate handlers when the same basicConfig() call is made in
multiple scripts.  Configuring once here and importing everywhere gives
consistent formatting, a single log file, and no duplicates.

Named 'logging_config' (not 'logging') to avoid shadowing the stdlib module.

Usage
-----
    from src.utils.logging_config import get_logger
    log = get_logger(__name__)
    log.info("Starting EEA download")
"""

import logging
import sys
from pathlib import Path

from src.utils.paths import LOGS_DIR

# ---------------------------------------------------------------------------
# Format  — timestamp | level | module name | message
# ---------------------------------------------------------------------------
_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger named `name` (pass __name__ from the calling module).

    On first call the root logger is configured with:
      - StreamHandler  → stdout (visible in terminal / Prefect UI)
      - FileHandler    → logs/pipeline.log (persistent, one file for the run)

    Subsequent calls return the already-configured logger for that module,
    so no duplicate handlers are added.
    """
    # Only configure handlers on the root logger once
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.INFO)

        formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

        # --- stdout handler ---------------------------------------------------
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

        # --- file handler -----------------------------------------------------
        log_file = LOGS_DIR / "pipeline.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return logging.getLogger(name)
