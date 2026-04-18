"""
retry.py
--------
A simple exponential-backoff retry decorator for network calls.

Why this exists
---------------
EEA's bulk download endpoint and the Copernicus CDS API are both known to
return transient errors (503, connection reset, timeout) under load.
Wrapping every download call in a manual try/except loop is repetitive and
inconsistent.  This decorator handles it once, cleanly, for the whole project.

Usage
-----
    from src.utils.retry import retry

    @retry(max_attempts=5, base_delay=10.0)
    def download_file(url, dest):
        ...
"""

import time
import functools
import logging
from typing import Tuple, Type

log = logging.getLogger(__name__)


def retry(
    max_attempts: int = 4,
    base_delay: float = 5.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator that retries the wrapped function on failure.

    Parameters
    ----------
    max_attempts : int
        Total number of attempts (1 = no retry).
    base_delay : float
        Seconds to wait before the first retry.
    backoff_factor : float
        Multiplier applied to the delay after each failure.
        With base_delay=5 and backoff_factor=2: waits 5s, 10s, 20s …
    exceptions : tuple of Exception types
        Only retry on these exception types.  Lets you skip retrying on,
        e.g., a 404 (file genuinely missing) vs a 503 (server overloaded).

    Example
    -------
        @retry(max_attempts=5, base_delay=10.0, exceptions=(requests.Timeout,))
        def fetch(url): ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        log.error(
                            "Function '%s' failed after %d attempts. "
                            "Last error: %s",
                            func.__name__, max_attempts, exc,
                        )
                        raise
                    log.warning(
                        "Attempt %d/%d for '%s' failed (%s). "
                        "Retrying in %.0fs …",
                        attempt, max_attempts, func.__name__, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator
