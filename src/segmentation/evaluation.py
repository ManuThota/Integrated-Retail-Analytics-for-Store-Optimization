"""
Segmentation Evaluation Module
================================
Evaluates the quality of K-Means store clusters using the Silhouette Score
and the Elbow Method (WCSS / inertia vs. k).

Notebook findings
──────────────────
  k=2 → Silhouette ≈ 0.42  (best mathematical separation)
  k=3 → Silhouette ≈ 0.25  (preferred for business insights)
  k=4 → Silhouette ≈ 0.25
  k=5 → Silhouette ≈ 0.26

The elbow in the WCSS curve occurs between k=2 and k=3, corroborating
the Silhouette analysis.
"""

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config.config import FIGURES_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)


def compute_silhouette(X_scaled: np.ndarray, labels: np.ndarray) -> float:
    """Compute the Silhouette Score for a given clustering assignment.

    A score close to +1 indicates dense, well-separated clusters.
    Around 0 suggests overlapping clusters.  Negative values mean a
    sample has been assigned to the wrong cluster.

    Args:
        X_scaled: Feature matrix used for clustering (same shape as
                  passed to KMeans).
        labels:   Cluster label array (e.g. ``kmeans.labels_``).

    Returns:
        Silhouette Score as a float in [-1, 1].
    """
    score = silhouette_score(X_scaled, labels)
    logger.info("Silhouette Score: %.4f", score)
    return score


def find_optimal_k(
    X_scaled: np.ndarray,
    k_range: range | None = None,
    random_state: int = RANDOM_STATE,
    n_init: int = 10,
) -> dict:
    """Evaluate multiple k values and return WCSS and Silhouette scores.

    Implements the Elbow Method (WCSS) together with Silhouette scoring
    so both criteria can inform the choice of k.

    Args:
        X_scaled:     Standardised feature matrix.
        k_range:      Range of k values to evaluate.  Default: ``range(2, 11)``.
        random_state: Reproducibility seed.
        n_init:       K-Means initialisations per k.

    Returns:
        Dictionary with keys:
          ``k_values``         – list of k values tested.
          ``wcss``             – list of inertia values (for Elbow plot).
          ``silhouette_scores``– list of Silhouette Scores per k.
    """
    k_range = k_range or range(2, 11)
    wcss        = []
    sil_scores  = []

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X_scaled)
        wcss.append(km.inertia_)
        sil    = silhouette_score(X_scaled, labels)
        sil_scores.append(sil)
        logger.info("k=%d — WCSS: %.2f | Silhouette: %.4f", k, km.inertia_, sil)

    return {
        "k_values":          list(k_range),
        "wcss":              wcss,
        "silhouette_scores": sil_scores,
    }


def plot_elbow_curve(evaluation: dict) -> None:
    """Save an Elbow / WCSS curve plot to ``reports/figures/``.

    Args:
        evaluation: Output of :func:`find_optimal_k`.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        evaluation["k_values"],
        evaluation["wcss"],
        marker="o",
        color="#4C72B0",
        linewidth=1.8,
    )
    ax.set_title("K-Means Elbow Method (WCSS vs k)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Within-Cluster Sum of Squares")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / "segmentation_elbow.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Elbow curve saved → %s", out)
