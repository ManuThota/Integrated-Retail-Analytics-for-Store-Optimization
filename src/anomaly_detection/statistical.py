"""
Statistical Anomaly Detection Module
======================================
Identifies outliers in retail sales and feature columns using
classical univariate statistical methods.

Two complementary approaches are provided:
  - **IQR (Interquartile Range)**: robust to heavy-tailed distributions;
    flags records that fall beyond ``k x IQR`` from Q1/Q3.
  - **Z-Score**: assumes approximate normality; flags records where the
    standardised value exceeds a given threshold (default ±3σ).

Both functions work on a specified column of a DataFrame and return a
boolean mask so callers can filter, count, or visualise anomalies freely.

Typical usage in a retail analytics context:
  - Detect weeks with abnormally high/low sales (data quality check).
  - Identify departments with extreme markdown spend.
  - Flag unusual spikes in CPI or Unemployment for external-factor analysis.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_iqr_outliers(
    df: pd.DataFrame,
    column: str,
    k: float = 1.5,
) -> pd.Series:
    """Detect outliers in *column* using the IQR fence method.

    A data point is flagged if it lies below ``Q1 − k·IQR`` or above
    ``Q3 + k·IQR``.  The standard Tukey value of ``k=1.5`` marks mild
    outliers; ``k=3.0`` marks only extreme outliers.

    Args:
        df:     Input DataFrame.
        column: Column name to analyse.  Must be numeric.
        k:      Fence multiplier (default 1.5 = standard Tukey fences).

    Returns:
        Boolean Series (True = outlier) aligned with *df*'s index.
    """
    if column not in df.columns:
        logger.warning("Column '%s' not found; returning empty outlier mask.", column)
        return pd.Series(False, index=df.index)

    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr     = q3 - q1
    lower   = q1 - k * iqr
    upper   = q3 + k * iqr

    mask = (df[column] < lower) | (df[column] > upper)
    n_outliers = mask.sum()

    logger.info(
        "IQR outlier detection on '%s': %d outliers found "
        "(lower=%.2f, upper=%.2f, k=%.1f).",
        column, n_outliers, lower, upper, k,
    )
    return mask


def detect_zscore_outliers(
    df: pd.DataFrame,
    column: str,
    threshold: float = 3.0,
) -> pd.Series:
    """Detect outliers in *column* using the Z-score method.

    A record is flagged when ``|z| > threshold`` where
    ``z = (x − mean) / std``.

    Note: Z-score is sensitive to the outliers themselves inflating the
    mean and std.  For heavily skewed data, prefer
    :func:`detect_iqr_outliers`.

    Args:
        df:        Input DataFrame.
        column:    Column name to analyse.  Must be numeric.
        threshold: Absolute Z-score threshold (default 3.0 = ±3σ).

    Returns:
        Boolean Series (True = outlier) aligned with *df*'s index.
    """
    if column not in df.columns:
        logger.warning("Column '%s' not found; returning empty outlier mask.", column)
        return pd.Series(False, index=df.index)

    z_scores = (df[column] - df[column].mean()) / df[column].std()
    mask      = z_scores.abs() > threshold
    n_outliers = mask.sum()

    logger.info(
        "Z-score outlier detection on '%s': %d outliers found "
        "(threshold=±%.1f).",
        column, n_outliers, threshold,
    )
    return mask


def summarise_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "iqr",
    **kwargs,
) -> pd.DataFrame:
    """Generate an outlier summary table for multiple columns at once.

    Args:
        df:      Input DataFrame.
        columns: List of column names to check.  Defaults to all numeric columns.
        method:  ``'iqr'`` or ``'zscore'``.
        **kwargs: Forwarded to the chosen detection function
                  (e.g. ``k=3.0`` for IQR or ``threshold=2.5`` for Z-score).

    Returns:
        DataFrame with columns: ``feature``, ``n_outliers``, ``pct_outliers``.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    detect_fn = detect_iqr_outliers if method == "iqr" else detect_zscore_outliers
    rows = []

    for col in columns:
        mask = detect_fn(df, col, **kwargs)
        rows.append({
            "feature":      col,
            "n_outliers":   int(mask.sum()),
            "pct_outliers": round(mask.mean() * 100, 2),
        })

    summary = pd.DataFrame(rows)
    logger.info(
        "Outlier summary (method='%s'): %d features checked.",
        method, len(columns),
    )
    return summary
