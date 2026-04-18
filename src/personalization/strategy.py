"""
Personalisation / Strategy Module
====================================
Translates K-Means cluster insights into actionable retail strategies
for each store segment.

This module bridges the gap between unsupervised learning outputs and
operational business decisions, answering the key question:
  "Given a store's cluster assignment, what actions should management take?"

Cluster business profiles (from notebook)
───────────────────────────────────────────
Cluster 1 – High-Performing Stores
  • Large stores, high sales, active markdown campaigns.
  • Strategy: Maintain performance; expand successful promotions.

Cluster 0 – Medium-Performing Stores
  • Mid-size stores, stable but improvable results.
  • Strategy: Targeted marketing campaigns; experiment with markdowns.

Cluster 2 – Low-Performing Stores
  • Small stores, low sales, minimal promotions.
  • Strategy: Deep-discount campaigns OR strategic resource reallocation.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

#: Business strategy templates indexed by cluster performance tier.
_STRATEGY_MAP = {
    "high": {
        "label":       "High-Performing",
        "strategies": [
            "Sustain and replicate current promotional mix across similar-sized stores.",
            "Invest in loyalty programmes to retain high-value customer segments.",
            "Pilot new product categories — high-traffic stores absorb risk well.",
            "Benchmark this cluster as the internal reference for store excellence.",
        ],
    },
    "medium": {
        "label":       "Medium-Performing",
        "strategies": [
            "Run targeted A/B tests on markdown depth and timing.",
            "Identify top-selling departments and allocate more floor space.",
            "Improve supply-chain responsiveness to reduce stockout frequency.",
            "Introduce localised promotions aligned with regional CPI trends.",
        ],
    },
    "low": {
        "label":       "Low-Performing",
        "strategies": [
            "Launch deep-discount campaigns to drive foot-traffic.",
            "Review and right-size product range to reduce holding costs.",
            "Evaluate whether store format (size, layout) is restricting growth.",
            "Consider closing underperforming departments and cross-selling.",
        ],
    },
}


def classify_clusters(cluster_summary: pd.DataFrame) -> dict:
    """Assign a performance tier (high / medium / low) to each cluster.

    Classification is based on ranked average Weekly_Sales within the
    cluster summary.

    Args:
        cluster_summary: Output of
            :func:`src.segmentation.clustering.get_cluster_summary`
            — a DataFrame indexed by Cluster with mean feature values.

    Returns:
        Dictionary mapping cluster label → performance tier string.
    """
    if "Weekly_Sales" not in cluster_summary.columns:
        logger.warning(
            "'Weekly_Sales' not in cluster summary; "
            "defaulting all clusters to 'medium'."
        )
        return {c: "medium" for c in cluster_summary.index}

    ranked  = cluster_summary["Weekly_Sales"].rank(ascending=True)
    n       = len(ranked)
    mapping = {}

    for cluster, rank in ranked.items():
        if rank == n:
            mapping[cluster] = "high"
        elif rank == 1:
            mapping[cluster] = "low"
        else:
            mapping[cluster] = "medium"

    logger.info("Cluster tier classification: %s", mapping)
    return mapping


def generate_store_strategies(
    cluster_summary: pd.DataFrame,
) -> str:
    """Generate a formatted strategy report for all store clusters.

    Args:
        cluster_summary: Output of
            :func:`src.segmentation.clustering.get_cluster_summary`.

    Returns:
        Multi-line strategy string for inclusion in ``reports/insights.txt``.
    """
    tier_map = classify_clusters(cluster_summary)
    lines    = [
        "=" * 60,
        "PERSONALISED STORE STRATEGIES BY CLUSTER",
        "=" * 60,
        "",
    ]

    for cluster_id, tier in tier_map.items():
        profile   = _STRATEGY_MAP[tier]
        lines += [
            f"  Cluster {cluster_id}  –  {profile['label']} Stores",
            "-" * 50,
        ]
        for i, action in enumerate(profile["strategies"], start=1):
            lines.append(f"    {i}. {action}")
        lines.append("")

    result = "\n".join(lines)
    logger.info("Strategy report generated for %d clusters.", len(tier_map))
    return result
