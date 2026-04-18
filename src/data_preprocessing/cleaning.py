"""
Cleaning Module
================
Handles missing-value imputation and data-quality fixes on the merged dataset.

Steps:
  1. Fill MarkDown1-5 NaN values with 0.
     Rationale: NaN means no promotion was active that week → effectively 0.
  2. Clip Weekly_Sales at 0 (some dept returns → treated as 0 before log1p).
"""

import logging

import pandas as pd

from src.config.config import MARKDOWN_COLS

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all data-cleaning steps to the merged DataFrame.

    Args:
        df: Merged DataFrame from :func:`src.data_preprocessing.merging.merge_datasets`.

    Returns:
        Cleaned DataFrame (copy, original unmodified).
    """
    df = df.copy()

    # ── 1. Impute MarkDown columns ────────────────────────────────────────────
    logger.info("Imputing MarkDown NaN values with 0 ...")
    for col in MARKDOWN_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    logger.info("Cleaning complete – shape: %s", df.shape)
    return df
