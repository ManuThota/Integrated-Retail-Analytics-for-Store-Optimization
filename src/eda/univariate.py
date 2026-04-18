"""
Univariate Analysis Module
===========================
Single-variable exploratory analysis for the retail dataset.

Generated outputs (saved to ``reports/``)
──────────────────────────────────────────
``reports/insights.txt``          – text EDA summary (shape, types, stats).
``reports/figures/eda_distributions.png``  – histograms of numeric features.
``reports/figures/eda_weekly_sales.png``   – distribution of raw sales values.

All plots are generated using a non-interactive Agg backend, making them
safe for server-side / headless environments.
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd

from src.config.config import REPORTS_DIR
from src.eda.visualization import ensure_figures_dir, save_fig

logger = logging.getLogger(__name__)


def write_text_summary(df: pd.DataFrame) -> None:
    """Write a comprehensive text-based dataset summary to
    ``reports/insights.txt``.

    Includes:
      - Dataset shape
      - Column data types
      - Missing value counts and percentages
      - Descriptive statistics (all dtypes)

    Args:
        df: The merged (raw) DataFrame before any transformation.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "insights.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("INTEGRATED RETAIL ANALYTICS — EDA SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"  Dataset Shape : {df.shape}\n\n")

        f.write("── Column Data Types ──────────────────────────────────────────────\n")
        f.write(df.dtypes.to_string())
        f.write("\n\n")

        f.write("── Missing Values ─────────────────────────────────────────────────\n")
        missing     = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df  = pd.DataFrame({"Count": missing, "Percentage (%)": missing_pct})
        non_zero    = missing_df[missing_df["Count"] > 0]
        if non_zero.empty:
            f.write("  No missing values found.\n")
        else:
            f.write(non_zero.to_string())
        f.write("\n\n")

        f.write("── Descriptive Statistics ─────────────────────────────────────────\n")
        f.write(df.describe(include="all").to_string())
        f.write("\n")

    logger.info("EDA text summary written → %s", output_path)


def plot_distributions(df: pd.DataFrame) -> None:
    """Histograms of key numeric columns → ``reports/figures/eda_distributions.png``.

    Args:
        df: The merged (raw) DataFrame.
    """
    ensure_figures_dir()
    numeric_cols = [
        c for c in ["Weekly_Sales", "Temperature", "Fuel_Price",
                     "CPI", "Unemployment", "Size"]
        if c in df.columns
    ]

    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):
        axes[idx].hist(df[col].dropna(), bins=40, edgecolor="white", color="#4C72B0")
        axes[idx].set_title(col, fontsize=11)
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Frequency")
        axes[idx].grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Distribution of Key Numeric Features", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "eda_distributions.png")


def plot_weekly_sales_distribution(df: pd.DataFrame) -> None:
    """Distribution of raw Weekly_Sales → ``reports/figures/eda_weekly_sales.png``.

    Args:
        df: The merged (raw) DataFrame.
    """
    ensure_figures_dir()
    if "Weekly_Sales" not in df.columns:
        logger.warning("'Weekly_Sales' not found — skipping sales distribution plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["Weekly_Sales"].dropna(), bins=60, edgecolor="white", color="#55A868")
    ax.set_title(
        "Distribution of Weekly Sales (Raw Values)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Weekly Sales ($)")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "eda_weekly_sales.png")
