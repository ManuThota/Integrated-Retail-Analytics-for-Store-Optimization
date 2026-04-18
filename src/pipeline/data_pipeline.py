"""
Data Pipeline Module
======================
Orchestrates Stages 1–4 of the end-to-end pipeline:

  Stage 1 – Data Ingestion   : Load raw CSVs from ``data/raw/``.
  Stage 2 – EDA              : Generate exploratory analysis reports.
  Stage 3 – Preprocessing    : Merge → clean → transform the dataset.
  Stage 4 – Checkpointing    : Persist interim and processed DataFrames.

Returns the fully preprocessed DataFrame so that
:func:`src.pipeline.train_pipeline.run_train_pipeline` can consume it
directly without re-reading from disk.
"""

import logging
import time

import pandas as pd

from src.data_ingestion.load_data import load_all_raw
from src.data_preprocessing.merging import merge_datasets
from src.data_preprocessing.cleaning import clean
from src.data_preprocessing.transformation import transform
from src.data_preprocessing.saving import save_interim, save_processed
from src.eda.univariate import (
    write_text_summary,
    plot_distributions,
    plot_weekly_sales_distribution,
)
from src.eda.bivariate import plot_sales_by_type, plot_holiday_effect
from src.eda.multivariate import plot_correlation_heatmap
from src.utils.helpers import format_duration

logger = logging.getLogger(__name__)

# ── Terminal banner helpers ───────────────────────────────────────────────────
_W = 58  # banner width


def _stage(n: int, total: int, msg: str) -> None:
    """Print a stage-start banner to the terminal."""
    print(f"\n  [ Stage {n}/{total} ]  ⟳  {msg}")


def _done(msg: str) -> None:
    """Print a stage-completion line to the terminal."""
    print(f"               ✓  {msg}")


def run_data_pipeline() -> pd.DataFrame:
    """Execute the data ingestion, EDA, and preprocessing stages.

    Returns:
        Fully preprocessed :class:`pandas.DataFrame` ready for feature
        engineering and model training.
    """
    # ── Stage 1: Data Ingestion ───────────────────────────────────────────────
    _stage(1, 8, "Loading raw datasets ...")
    t0 = time.time()
    sales_df, stores_df, features_df = load_all_raw()
    _done(
        f"Loaded  sales={sales_df.shape[0]:,} rows  |  "
        f"stores={stores_df.shape[0]:,}  |  "
        f"features={features_df.shape[0]:,}  "
        f"({format_duration(time.time() - t0)})"
    )

    # ── Merge for EDA (single merge — reused for preprocessing too) ───────────
    merged_raw = merge_datasets(sales_df, stores_df, features_df)

    # ── Stage 2: EDA ──────────────────────────────────────────────────────────
    _stage(2, 8, "Running EDA  (summary + 5 plots) ...")
    t0 = time.time()
    write_text_summary(merged_raw)
    plot_distributions(merged_raw)
    plot_weekly_sales_distribution(merged_raw)
    plot_sales_by_type(merged_raw)
    plot_holiday_effect(merged_raw)
    plot_correlation_heatmap(merged_raw)
    _done(
        f"EDA complete — insights.txt + 5 plots saved to reports/figures/  "
        f"({format_duration(time.time() - t0)})"
    )

    # ── Stage 3: Preprocessing ────────────────────────────────────────────────
    _stage(3, 8, "Preprocessing  (clean → log1p transform) ...")
    t0 = time.time()
    df_clean       = clean(merged_raw)
    df_transformed = transform(df_clean)
    _done(
        f"Preprocessing complete — shape: {df_transformed.shape}  "
        f"({format_duration(time.time() - t0)})"
    )

    # ── Stage 4: Save checkpoints ──────────────────────────────────────────────
    _stage(4, 8, "Saving data checkpoints ...")
    t0 = time.time()
    save_interim(merged_raw)        # → data/interim/merged_data.csv
    save_processed(df_transformed)  # → data/processed/final_data.csv
    _done("data/interim/merged_data.csv")
    _done(
        f"data/processed/final_data.csv  "
        f"({format_duration(time.time() - t0)})"
    )

    return df_transformed
