"""
External Factors Analysis Module
===================================
Quantifies the impact of macroeconomic variables on weekly retail sales.

External factors in this dataset
──────────────────────────────────
CPI (Consumer Price Index)
    Higher CPI → higher cost of living → potential reduction in discretionary
    retail spending.

Unemployment Rate
    Higher unemployment → lower disposable income → reduced retail demand.

Fuel Price
    Higher fuel prices → increased travel cost → fewer in-store visits
    (primarily relevant for large-format stores like Walmart).

Analysis approach
──────────────────
Pearson correlation gives the linear association between each external
factor and weekly sales.  Grouping by store segments (from K-Means) reveals
whether large vs. small stores are differently affected.

All results are logged and optionally returned as DataFrames for further
reporting or plotting.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

#: External factor columns available in the Walmart retail dataset.
EXTERNAL_COLS = ["CPI", "Unemployment", "Fuel_Price"]


def compute_factor_correlations(
    df: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    factor_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate Pearson correlation between each external factor and sales.

    Args:
        df:          DataFrame containing sales and factor columns.
        target_col:  Sales column name (default ``Weekly_Sales``).
        factor_cols: External factor columns to analyse.
                     Defaults to ``['CPI', 'Unemployment', 'Fuel_Price']``.

    Returns:
        DataFrame with columns ``factor``, ``pearson_r``, ``interpretation``
        sorted by absolute correlation (strongest first).
    """
    factor_cols = factor_cols or EXTERNAL_COLS
    present     = [c for c in factor_cols if c in df.columns]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    rows = []
    for col in present:
        r = df[col].corr(df[target_col], method="pearson")
        rows.append({
            "factor":         col,
            "pearson_r":     round(r, 4),
            "interpretation": _interpret_r(r),
        })

    result = (
        pd.DataFrame(rows)
        .sort_values("pearson_r", key=abs, ascending=False)
        .reset_index(drop=True)
    )

    for _, row in result.iterrows():
        logger.info(
            "Factor correlation — %s vs %s: r = %.4f  (%s)",
            row["factor"], target_col, row["pearson_r"], row["interpretation"],
        )
    return result


def analyse_by_store_cluster(
    df: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    factor_cols: list[str] | None = None,
    cluster_col: str = "Cluster",
) -> dict[str, pd.DataFrame]:
    """Compute factor correlations separately for each store cluster.

    Useful for understanding whether external conditions impact
    high-performing and low-performing stores differently.

    Args:
        df:          DataFrame with sales, factors, and a cluster column.
        target_col:  Target sales column.
        factor_cols: External factors to analyse.
        cluster_col: Column containing cluster assignments.

    Returns:
        Dictionary mapping cluster label → correlation DataFrame.
    """
    factor_cols = factor_cols or EXTERNAL_COLS

    if cluster_col not in df.columns:
        logger.warning(
            "'%s' column not found.  Run segmentation first.", cluster_col
        )
        return {}

    results = {}
    for cluster_id, grp in df.groupby(cluster_col):
        results[cluster_id] = compute_factor_correlations(
            grp, target_col=target_col, factor_cols=factor_cols
        )
        logger.info("External factor analysis complete for Cluster %s.", cluster_id)

    return results


def _interpret_r(r: float) -> str:
    """Map a Pearson r value to a human-readable strength label."""
    abs_r = abs(r)
    direction = "positive" if r > 0 else "negative"
    if abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.3:
        strength = "moderate"
    else:
        strength = "weak"
    return f"weak {direction}" if strength == "weak" else f"{strength} {direction}"
