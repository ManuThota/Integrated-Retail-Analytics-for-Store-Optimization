"""
Logger Utility
==============
Centralised logging configuration for the entire project.

Configures the ROOT logger so every module that calls
``logging.getLogger(__name__)`` automatically inherits the console
and file handlers — no additional setup required in sub-modules.

Log output goes to:
  - stdout (console)                 ← visible in the terminal
  - logs/pipeline.log (rotating file, path from config.py)

Usage (from any module):
    import logging
    logger = logging.getLogger(__name__)
    logger.info("This will show in the terminal.")

Or to initialise root logging explicitly from main.py:
    from src.utils.logger import get_logger
    get_logger(__name__)
"""

import logging
import sys

from src.config.config import LOG_FILE


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure the ROOT logger once and return a named child logger.

    Configuring the root logger means every module's
    ``logging.getLogger(__name__)`` automatically inherits the console
    and file handlers through Python's logging propagation chain.

    Args:
        name:  Typically ``__name__`` of the calling module.
        level: Logging level (default INFO).

    Returns:
        Configured :class:`logging.Logger` for *name*.
    """
    root = logging.getLogger()  # ← the true root logger

    # Only set up handlers once to avoid duplicate log entries
    if not root.handlers:
        root.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # ── Console handler (stdout) ──────────────────────────────────────────
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        root.addHandler(ch)

        # ── File handler ──────────────────────────────────────────────────────
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    return logging.getLogger(name)
