"""
Multivariate Analysis Module
=============================
Multi-variable analysis — relationships across three or more dimensions.

Generated outputs
──────────────────
``reports/figures/eda_correlation.png``  – Pearson correlation heatmap of all
                                           numeric columns in the merged dataset.

The heatmap is most useful on the merged *raw* DataFrame (before log1p)
so the correlations reflect original-scale relationships that are easier
to interpret against business intuition (e.g., larger store → higher sales).
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.eda.visualization import ensure_figures_dir, save_fig

logger = logging.getLogger(__name__)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Pearson correlation heatmap for all numeric columns
    → ``reports/figures/eda_correlation.png``.

    Only numeric columns are included (object / category columns are
    automatically excluded by :meth:`pandas.DataFrame.select_dtypes`).

    Args:
        df: The merged (raw) DataFrame.
    """
    ensure_figures_dir()
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        logger.warning("No numeric columns found — skipping correlation heatmap.")
        return

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=True,
        ax=ax,
    )
    ax.set_title("Pearson Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "eda_correlation.png")
