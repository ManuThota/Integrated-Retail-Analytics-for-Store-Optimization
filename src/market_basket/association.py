"""
Market Basket / Department Association Module
===============================================
Identifies store-department co-performance patterns — i.e., which
departments tend to generate high (or low) sales *together* across stores.

Retail analytics context
──────────────────────────
Traditional Market Basket Analysis (Apriori) works on transaction-level
item co-purchases.  The Walmart retail dataset operates at weekly
store-department granularity, so the equivalent analysis is:

  **"Which departments reliably co-perform at the same stores?"**

This is answered via Pearson correlation of department-level aggregated
sales across stores:  a high correlation between Dept A and Dept B means
stores with strong Dept A performance also tend to have strong Dept B,
enabling coordinated inventory, staffing, and promotional decisions.

Functions
──────────
``build_dept_pivot``       – pivot weekly_sales to (Store × Dept) matrix.
``compute_dept_correlation``– pairwise Pearson correlation between depts.
``find_top_pairs``         – surface the most positively correlated dept pairs.
``summarise_basket``       – formatted text summary of top pairs.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_dept_pivot(df: pd.DataFrame, sales_col: str = "Weekly_Sales") -> pd.DataFrame:
    """Pivot the dataset to a (Store × Dept) average-sales matrix.

    Each cell ``[store, dept]`` contains the mean weekly sales for that
    store-department combination.  This makes departments directly
    comparable across stores.

    Args:
        df:        DataFrame with at least ``Store``, ``Dept``, and a sales col.
        sales_col: Column to aggregate (default ``Weekly_Sales``).

    Returns:
        Pivot DataFrame: rows = stores, columns = departments.
    """
    pivot = df.pivot_table(
        index="Store",
        columns="Dept",
        values=sales_col,
        aggfunc="mean",
    )
    logger.info(
        "Department pivot: %d stores × %d departments.", pivot.shape[0], pivot.shape[1]
    )
    return pivot


def compute_dept_correlation(pivot: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Pearson correlation between departments.

    Args:
        pivot: Output of :func:`build_dept_pivot` (Store × Dept matrix).

    Returns:
        Square correlation matrix (Dept × Dept).
    """
    corr = pivot.corr(method="pearson")
    logger.info("Department correlation matrix computed (%d depts).", len(corr))
    return corr


def find_top_pairs(
    corr_matrix: pd.DataFrame,
    top_n: int = 10,
    threshold: float = 0.7,
) -> pd.DataFrame:
    """Extract the top-N most positively correlated department pairs.

    Args:
        corr_matrix: Department correlation matrix from
                     :func:`compute_dept_correlation`.
        top_n:       Number of top pairs to return (default 10).
        threshold:   Minimum correlation strength to include (default 0.7).

    Returns:
        DataFrame with columns ``dept_a``, ``dept_b``, ``correlation``
        sorted descending by correlation strength.
    """
    # Stack the upper triangle only (avoid duplicates and self-correlations)
    pairs = (
        corr_matrix
        .where(
            # upper triangle mask
            pd.DataFrame(data=True, index=corr_matrix.index, columns=corr_matrix.columns)
            .where(
                lambda df: df.apply(lambda x: [i > j for i, j in zip(range(len(x)), range(len(x)))])
            )
        )
        .stack()
        .reset_index()
    )
    pairs.columns = ["dept_a", "dept_b", "correlation"]
    pairs = (
        pairs[pairs["correlation"] >= threshold]
        .sort_values("correlation", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    logger.info(
        "Top %d department pairs found (threshold=%.2f).", len(pairs), threshold
    )
    return pairs


def summarise_basket(top_pairs: pd.DataFrame) -> str:
    """Format a human-readable summary of the top correlated department pairs.

    Args:
        top_pairs: Output of :func:`find_top_pairs`.

    Returns:
        Formatted multi-line string suitable for writing to ``insights.txt``.
    """
    lines = [
        "=" * 55,
        "DEPARTMENT CO-PERFORMANCE ANALYSIS (Market Basket)",
        "=" * 55,
        "",
        "Top strongly correlated department pairs:",
        "  (High correlation = tend to perform well together)",
        "",
    ]
    for _, row in top_pairs.iterrows():
        lines.append(
            f"  Dept {int(row['dept_a']):>3}  ↔  Dept {int(row['dept_b']):>3}"
            f"   Correlation: {row['correlation']:.4f}"
        )

    result = "\n".join(lines)
    logger.info("Market basket summary generated for %d pairs.", len(top_pairs))
    return result
