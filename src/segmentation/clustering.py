"""
K-Means Clustering Module
===========================
Segments retail stores into performance-based clusters using K-Means.

Notebook findings (replicated here)
─────────────────────────────────────
After aggregating sales, markdowns, and macroeconomic indicators at the
store level, K-Means with ``k=3`` produced interpretable segments:

  Cluster 1 – High-Performing Stores
    • High average weekly sales
    • Large store size
    • High markdown investment (strong promotional presence)

  Cluster 0 – Medium-Performing Stores
    • Moderate sales and promotions
    • Mid-range store size
    • Stable, improvable operations

  Cluster 2 – Low-Performing Stores
    • Low sales and minimal markdowns
    • Small stores
    • Require targeted strategy review

Silhouette scores:
  k=2 → ~0.42  (strongest separation)
  k=3 → ~0.25  (more business-useful segments despite lower cohesion)

``k=3`` is used by default for richer business insights.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config.config import CLUSTER_FEATURE_COLS, RANDOM_STATE

logger = logging.getLogger(__name__)


def prepare_cluster_data(df: pd.DataFrame) -> tuple:
    """Aggregate the full dataset to one row per store and scale features.

    Computes per-store means of sales, markdowns, CPI, and unemployment;
    ``Size`` is taken as ``first`` (it is constant per store).

    Args:
        df: Merged DataFrame (pre- or post-transformation).
            Must contain a ``Store`` column and the columns listed in
            ``CLUSTER_FEATURE_COLS`` (config).

    Returns:
        Tuple ``(df_cluster, X_scaled)`` where:
          - ``df_cluster``  – store-level aggregated DataFrame.
          - ``X_scaled``    – StandardScaler-normalised numpy array
                              ready for K-Means.
    """
    # Build aggregation dictionary; Size is constant per store, use first
    agg = {}
    for col in CLUSTER_FEATURE_COLS:
        if col in df.columns:
            agg[col] = "first" if col == "Size" else "mean"

    if not agg:
        raise ValueError(
            "None of the CLUSTER_FEATURE_COLS found in the DataFrame. "
            "Check config.py → CLUSTER_FEATURE_COLS."
        )

    df_cluster = df.groupby("Store").agg(agg).reset_index()
    feature_cols = [c for c in CLUSTER_FEATURE_COLS if c in df_cluster.columns]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster[feature_cols])

    logger.info(
        "Cluster data prepared: %d stores × %d features.",
        len(df_cluster), len(feature_cols),
    )
    return df_cluster, X_scaled


def run_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = 3,
    random_state: int = RANDOM_STATE,
    n_init: int = 10,
) -> KMeans:
    """Fit a K-Means model on the scaled store feature matrix.

    Args:
        X_scaled:    Standardised feature matrix (output of
                     :func:`prepare_cluster_data`).
        n_clusters:  Number of clusters (default 3, per notebook analysis).
        random_state: Seed for reproducibility.
        n_init:      Number of K-Means initialisations (default 10).

    Returns:
        Fitted :class:`~sklearn.cluster.KMeans` with ``labels_`` populated.
    """
    logger.info(
        "Fitting K-Means (k=%d, n_init=%d, random_state=%d) ...",
        n_clusters, n_init, random_state,
    )
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    km.fit(X_scaled)
    logger.info("K-Means fit complete. Inertia: %.4f", km.inertia_)
    return km


def get_cluster_summary(df_cluster: pd.DataFrame) -> pd.DataFrame:
    """Compute per-cluster mean statistics for business interpretation.

    Args:
        df_cluster: Store-level DataFrame with a ``Cluster`` column
                    added by :func:`run_kmeans`.

    Returns:
        DataFrame indexed by ``Cluster`` with mean values per feature.
    """
    if "Cluster" not in df_cluster.columns:
        raise ValueError("'Cluster' column missing — run run_kmeans() first.")

    summary_cols = [c for c in CLUSTER_FEATURE_COLS if c in df_cluster.columns]
    summary = df_cluster.groupby("Cluster")[summary_cols].mean()

    logger.info(
        "Cluster summary: %d clusters, %d features.", len(summary), len(summary_cols)
    )
    return summary
