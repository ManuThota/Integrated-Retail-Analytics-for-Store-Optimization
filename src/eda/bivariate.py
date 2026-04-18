"""
Bivariate Analysis Module
==========================
Two-variable exploratory analysis — how does one variable relate to another?

Generated outputs (saved to ``reports/figures/``)
───────────────────────────────────────────────────
``eda_sales_by_type.png``    – box-plots of Weekly_Sales by Store Type.
``eda_holiday_effect.png``   – bar chart: average sales holiday vs non-holiday.

Design note
────────────
Both functions accept the *raw merged* DataFrame (before transformation)
so they display original dollar-scale sales values and human-readable
store type labels (A / B / C).
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.eda.visualization import ensure_figures_dir, save_fig

logger = logging.getLogger(__name__)


def plot_sales_by_type(df: pd.DataFrame) -> None:
    """Box-plots of Weekly_Sales grouped by Store Type
    → ``reports/figures/eda_sales_by_type.png``.

    Handles both raw string Type (A / B / C) and integer-encoded Type
    (0 / 1 / 2) so it can be called on either pre- or post-encoded data.

    Args:
        df: Merged DataFrame (preferably raw, before OHE).
    """
    ensure_figures_dir()
    if "Type" not in df.columns or "Weekly_Sales" not in df.columns:
        logger.warning("'Type' or 'Weekly_Sales' missing — skipping type plot.")
        return

    # Map integer codes back to readable labels if OHE was already applied
    int_to_label = {0: "A", 1: "B", 2: "C"}
    plot_df = df[["Type", "Weekly_Sales"]].copy()

    if plot_df["Type"].dtype != object:
        plot_df["StoreType"] = (
            plot_df["Type"].map(int_to_label).fillna(plot_df["Type"].astype(str))
        )
    else:
        plot_df["StoreType"] = plot_df["Type"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=plot_df,
        x="StoreType",
        y="Weekly_Sales",
        hue="StoreType",
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.set_title(
        "Weekly Sales Distribution by Store Type",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Store Type")
    ax.set_ylabel("Weekly Sales ($)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "eda_sales_by_type.png")


def plot_holiday_effect(df: pd.DataFrame) -> None:
    """Bar chart: average Weekly_Sales in holiday vs. non-holiday weeks
    → ``reports/figures/eda_holiday_effect.png``.

    Args:
        df: Merged DataFrame (raw or preprocessed).
    """
    ensure_figures_dir()
    if "IsHoliday" not in df.columns or "Weekly_Sales" not in df.columns:
        logger.warning("'IsHoliday' or 'Weekly_Sales' missing — skipping holiday plot.")
        return

    avg = (
        df.groupby("IsHoliday")["Weekly_Sales"]
        .mean()
        .reset_index()
    )
    avg["Label"] = avg["IsHoliday"].map({True: "Holiday", False: "Non-Holiday"})

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(
        avg["Label"],
        avg["Weekly_Sales"],
        color=["#4C72B0", "#DD8452"],
        edgecolor="white",
        width=0.5,
    )
    ax.set_title(
        "Average Weekly Sales: Holiday vs Non-Holiday",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Week Type")
    ax.set_ylabel("Average Weekly Sales ($)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "eda_holiday_effect.png")
