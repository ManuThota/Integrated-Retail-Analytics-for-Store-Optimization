"""
Feature Builder Module
=======================
Constructs derived (engineered) features from the raw / cleaned dataset
beyond what simple column selects provide.

Engineering rationale (from notebook analysis)
────────────────────────────────────────────────
* **Temporal features** (Year / Month / Week): the date of a shopping week
  is a strong proxy for seasonal demand and macroeconomic conditions.
* **Markdown aggregates** (TotalMarkDown / ActiveMarkDowns): raw MarkDown1-5
  values are sparse; aggregating them captures total promotional pressure and
  the *count* of active promotions — two distinct business signals.

These features are applied to the *cleaned* but un-transformed DataFrame
(before log1p of Weekly_Sales) so they operate on the original scale.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

#: MarkDown columns present in the Walmart retail dataset.
_MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract ``Year``, ``Month``, and ISO calendar ``Week`` from the
    ``Date`` column.

    The raw date string (DD-MM-YYYY) is parsed with ``dayfirst=True`` to
    avoid month/day confusion.  The resulting temporal columns give the
    Random Forest direct access to seasonal and yearly patterns.

    Args:
        df: DataFrame containing a ``Date`` column.

    Returns:
        Copy of *df* with ``Year``, ``Month``, and ``Week`` columns added.
        The raw ``Date`` column is **retained** here; dropping it is left to
        :func:`src.data_preprocessing.transformation.transform`.
    """
    df = df.copy()

    if "Date" not in df.columns:
        logger.warning("'Date' column not found; skipping time feature extraction.")
        return df

    logger.info("Building temporal features (Year, Month, Week) from Date ...")
    df["Date"]  = pd.to_datetime(df["Date"], dayfirst=True)
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"]  = df["Date"].dt.isocalendar().week.astype(int)

    return df


def build_markdown_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the five MarkDown columns into two synthetic features.

    The individual MarkDown1-5 values capture *which* promotion type was
    applied, but the model also benefits from knowing:
      - ``TotalMarkDown``  : total promotional investment in a given week.
      - ``ActiveMarkDowns``: number of distinct promotion types active.

    NaN values are treated as 0 (no promotion) before aggregation, which
    is consistent with the cleaning step in ``cleaning.py``.

    Args:
        df: DataFrame containing MarkDown1-5 columns (may contain NaNs).

    Returns:
        Copy of *df* with two new columns: ``TotalMarkDown`` and
        ``ActiveMarkDowns``.
    """
    df = df.copy()
    available_md = [c for c in _MARKDOWN_COLS if c in df.columns]

    if not available_md:
        logger.warning("No MarkDown columns found; skipping markdown feature build.")
        return df

    logger.info(
        "Building markdown aggregate features from %d MarkDown columns ...",
        len(available_md),
    )
    md_df = df[available_md].fillna(0)

    df["TotalMarkDown"]   = md_df.sum(axis=1)
    df["ActiveMarkDowns"] = (md_df > 0).sum(axis=1)

    return df
