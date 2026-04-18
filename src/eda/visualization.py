"""
EDA Visualization Helpers
===========================
Shared utilities for all EDA sub-modules in ``src/eda/``.

All generated plots are saved to ``reports/figures/`` (defined as
``FIGURES_DIR`` in ``config.py``) to match the project's reports structure::

    reports/
    ├── figures/       ← all .png plots
    ├── metrics.json   ← model evaluation metrics
    └── insights.txt   ← human-readable text summary

Usage (from any EDA module)::

    from src.eda.visualization import ensure_figures_dir, save_fig
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless / server-safe backend — no display required
import matplotlib.pyplot as plt

from src.config.config import FIGURES_DIR

logger = logging.getLogger(__name__)


def ensure_figures_dir() -> None:
    """Create ``reports/figures/`` and its parents if they do not yet exist.

    Idempotent — safe to call multiple times.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, filename: str) -> Path:
    """Save a matplotlib Figure to ``reports/figures/`` and close it.

    Closing the figure immediately after saving prevents memory leaks when
    many plots are generated in a single pipeline run.

    Args:
        fig:      The :class:`matplotlib.figure.Figure` to save.
        filename: Output filename (e.g. ``'eda_distributions.png'``).
                  Must include the file extension.

    Returns:
        :class:`pathlib.Path` pointing to the saved file.
    """
    ensure_figures_dir()
    path = FIGURES_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Plot saved → %s", path)
    return path
