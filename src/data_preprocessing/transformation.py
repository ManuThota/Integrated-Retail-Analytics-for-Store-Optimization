"""
Transformation Module
======================
Applies feature-level transformations to the cleaned, merged DataFrame.

Pipeline steps
──────────────
1. Parse the ``Date`` column (stored as DD-MM-YYYY) → datetime.
2. Extract temporal features: ``Year``, ``Month``, ``Week``.
3. Apply ``log1p`` to ``Weekly_Sales`` to reduce right-skew of the target.
4. Drop the raw ``Date`` column (rendered redundant by step 2).

**What this module does NOT do**
─────────────────────────────────
- It does NOT encode the ``Type`` column.
  The original notebook used **One-Hot Encoding** for the store-type
  categorical variable (A / B / C).  That step is deliberately deferred to
  ``src/feature_engineering/encoding.py`` so that:
    • The processed CSV (``data/processed/final_data.csv``) stores the raw
      human-readable labels (A / B / C) for analysis and EDA re-use.
    • OHE is applied in-memory during the model training phase only.

- It does NOT clip or impute missing values (handled in ``cleaning.py``).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all transformation steps to the cleaned, merged DataFrame.

    The function operates on a copy of the input so the original is never
    mutated.

    Args:
        df: Cleaned DataFrame from :func:`src.data_preprocessing.cleaning.clean`.

    Returns:
        Transformed DataFrame with:
          - ``Year``, ``Month``, ``Week`` columns added.
          - ``Weekly_Sales`` replaced by its ``log1p``-transformed values.
          - Raw ``Date`` column removed.
          - ``Type`` column **unchanged** (still 'A' / 'B' / 'C') — OHE
            is applied later in the feature-engineering stage.
    """
    df = df.copy()

    # ── Step 1 & 2: Date parsing → temporal feature extraction ───────────────
    # CSV dates are formatted as DD-MM-YYYY; dayfirst=True ensures that
    # 05-02-2010 is read as 5 Feb 2010, not 2 May 2010.
    logger.info("Parsing Date column (DD-MM-YYYY format) ...")
    df["Date"]  = pd.to_datetime(df["Date"], dayfirst=True)
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    # isocalendar().week returns a UInt32 Series; cast to int for sklearn
    df["Week"]  = df["Date"].dt.isocalendar().week.astype(int)

    # ── Step 3: Log-transform the target variable ─────────────────────────────
    # Some department returns are negative (markdown-heavy weeks).
    # Clipping at 0 before log1p prevents NaN / -inf while preserving the
    # monotonic ordering of positive sales values.
    logger.info("Applying log1p transformation to Weekly_Sales ...")
    df["Weekly_Sales"] = np.log1p(df["Weekly_Sales"].clip(lower=0))

    # ── Step 4: Drop the raw Date column ─────────────────────────────────────
    # Year / Month / Week carry all the temporal signal the model needs.
    logger.info("Dropping raw Date column ...")
    df.drop(columns=["Date"], inplace=True, errors="ignore")

    logger.info(
        "Transformation complete – shape: %s  |  'Type' column preserved as "
        "string for downstream One-Hot Encoding.",
        df.shape,
    )
    return df
