"""Logging setup shared by every pipeline stage.

Logs go to stderr and, when `log_dir` is given, to a rotating file. The format includes the
stage name so that logs from a scheduled job are greppable per stage.
"""
from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_dir: str | Path | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(_FMT))
    root.addHandler(console)
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(Path(log_dir) / "m5.log", maxBytes=20_000_000, backupCount=3)
        fh.setFormatter(logging.Formatter(_FMT))
        root.addHandler(fh)
    # third-party noise
    for noisy in ("numba", "pytensor", "arviz", "arviz_stats", "arviz_base"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@contextmanager
def timed(logger: logging.Logger, msg: str) -> Iterator[None]:
    """Log how long a block took, also on failure."""
    t0 = time.perf_counter()
    logger.info("START %s", msg)
    try:
        yield
    except Exception:
        logger.exception("FAILED %s after %.1fs", msg, time.perf_counter() - t0)
        raise
    logger.info("DONE  %s in %.1fs", msg, time.perf_counter() - t0)
