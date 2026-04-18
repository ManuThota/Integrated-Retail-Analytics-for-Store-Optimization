"""
Feature Importance & Explainability Module
============================================
Generates feature importance plots and analysis for the trained
Random Forest Regressor.

Two importance measures are available:

1. **MDI (Mean Decrease in Impurity)** — built into scikit-learn RF.
   Fast to compute; can over-rank high-cardinality numerical features.

2. **Permutation Importance** — model-agnostic, computed by randomly
   shuffling one feature at a time and measuring the drop in R².
   Slower but more robust and statistically meaningful.

The notebook relied on MDI for quick feature ranking during EDA.
Permutation Importance is provided here for deeper analysis.

Both measures are saved to ``reports/figures/`` and a ranked text
summary is written to ``reports/insights.txt``.
"""

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from src.config.config import FIGURES_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)


def plot_mdi_importance(
    model: RandomForestRegressor,
    feature_names: list[str],
    top_n: int = 15,
) -> pd.Series:
    """Plot Mean Decrease in Impurity (MDI) feature importances.

    MDI is extracted directly from the fitted RF as
    ``model.feature_importances_``.  Top *top_n* features are displayed
    in a horizontal bar chart.

    Args:
        model:         Fitted :class:`~sklearn.ensemble.RandomForestRegressor`.
        feature_names: Ordered list of feature column names matching X_train.
        top_n:         Number of top features to display (default 15).

    Returns:
        Sorted :class:`pandas.Series` of importances (descending).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_feats   = importances.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_feats.plot(kind="barh", color="#4C72B0", edgecolor="white", ax=ax)
    ax.set_title(
        f"Top {top_n} Feature Importances (MDI) – Random Forest",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / "feature_importance_mdi.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("MDI importance plot saved → %s", out)

    return importances.sort_values(ascending=False)


def compute_permutation_importance(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute Permutation Importance on the held-out test set.

    Each feature is shuffled *n_repeats* times.  The mean R² drop across
    repeats gives a robust, model-agnostic importance estimate.

    Args:
        model:        Fitted Random Forest.
        X_test:       Scaled test feature matrix.
        y_test:       Log1p-transformed test target.
        n_repeats:    Number of permutations per feature (default 10).
        random_state: Reproducibility seed.

    Returns:
        DataFrame with columns ``feature``, ``importance_mean``,
        ``importance_std``, sorted by mean importance descending.
    """
    logger.info(
        "Computing permutation importance (%d repeats, %d features) ...",
        n_repeats, X_test.shape[1],
    )

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring="r2",
    )

    perm_df = pd.DataFrame({
        "feature":         X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std":  result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    logger.info(
        "Top feature by permutation importance: %s (mean R² drop = %.4f)",
        perm_df.iloc[0]["feature"],
        perm_df.iloc[0]["importance_mean"],
    )
    return perm_df


def save_importance_report(importances: pd.Series) -> None:
    """Append the MDI importance ranking to ``reports/insights.txt``.

    Args:
        importances: Sorted Series from :func:`plot_mdi_importance`.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "insights.txt"

    with open(out, "a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write("=" * 55 + "\n")
        f.write("FEATURE IMPORTANCE RANKING (MDI)\n")
        f.write("=" * 55 + "\n\n")
        for rank, (feat, imp) in enumerate(importances.items(), start=1):
            f.write(f"  {rank:>2}. {feat:<24} {imp:.6f}\n")

    logger.info("Feature importance report appended → %s", out)
