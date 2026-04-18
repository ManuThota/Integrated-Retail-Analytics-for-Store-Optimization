"""
Hyperparameter Tuning Module
==============================
Provides cross-validation scoring and hyperparameter optimisation for the
Random Forest Regressor using scikit-learn's search utilities.

Two search strategies are implemented:

``tune_with_randomized_search`` (recommended first step)
    Randomly samples ``n_iter`` combinations from the parameter space —
    fast exploration of a broad search space.  Uses the distribution defined
    in ``PARAM_DISTRIBUTIONS`` (config.py).

``tune_with_grid_search`` (optional refinement step)
    Exhaustively evaluates every combination in a focused grid — ideal as a
    follow-up after Randomized Search has identified a promising region.
    Uses ``FINE_TUNE_GRID`` (config.py) by default.

Cross-validation (``run_cross_validation``)
    Standalone k-fold CV to assess a model's generalisation before committing
    to a full tuning run.  Matches the notebook's reported CV R² ≈ 0.968.

Recommended workflow
─────────────────────
1. ``run_cross_validation``         → confirm baseline R² on training data.
2. ``tune_with_randomized_search``  → quickly identify best parameter region.
3. ``tune_with_grid_search``        → (optional) fine-tune on a narrow grid.
4. Feed ``best_estimator_`` into ``train_pipeline.py`` for final fit.

All results are persisted to ``reports/`` for reproducibility.
"""

import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)

from src.config.config import (
    CV_FOLDS,
    FINE_TUNE_GRID,
    N_ESTIMATORS,
    N_ITER_SEARCH,
    PARAM_DISTRIBUTIONS,
    RANDOM_STATE,
    REPORTS_DIR,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Cross-Validation
# ══════════════════════════════════════════════════════════════════════════════

def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = CV_FOLDS,
    scoring: str = "r2",
    n_estimators: int = N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Run k-fold cross-validation on the Random Forest Regressor.

    A fresh, unfitted RF is created internally so each fold is evaluated
    independently.  ``n_jobs=-1`` uses all available CPU cores for
    parallel fold evaluation.

    Args:
        X:            Feature matrix (training split only — avoids leakage).
        y:            Target series (log1p-transformed Weekly_Sales).
        cv:           Number of folds.  Default: ``CV_FOLDS`` = 5 (config).
        scoring:      scikit-learn scoring metric.  Default: ``'r2'``.
        n_estimators: Trees per fold's RF.  Default: ``N_ESTIMATORS`` = 100.
        random_state: Reproducibility seed.

    Returns:
        Dictionary with keys:
          ``cv_scores``  – numpy array of per-fold R² scores.
          ``mean``       – mean R² across all folds.
          ``std``        – standard deviation across folds.
          ``cv_folds``   – number of folds used.
          ``scoring``    – scoring metric used.
          ``elapsed_s``  – wall-clock seconds for the CV run.
    """
    logger.info(
        "Running %d-fold cross-validation (scoring='%s', n_estimators=%d) ...",
        cv, scoring, n_estimators,
    )

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    t0        = time.time()
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    elapsed   = round(time.time() - t0, 2)

    result = {
        "cv_scores": cv_scores,
        "mean":      float(np.mean(cv_scores)),
        "std":       float(np.std(cv_scores)),
        "cv_folds":  cv,
        "scoring":   scoring,
        "elapsed_s": elapsed,
    }

    logger.info(
        "Cross-validation done in %.1fs — Mean %s: %.4f ± %.4f",
        elapsed, scoring.upper(), result["mean"], result["std"],
    )
    _save_cv_results(result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameter Tuning
# ══════════════════════════════════════════════════════════════════════════════

def tune_with_randomized_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_distributions: dict | None = None,
    n_iter: int = N_ITER_SEARCH,
    cv: int = CV_FOLDS,
    scoring: str = "r2",
    random_state: int = RANDOM_STATE,
) -> RandomizedSearchCV:
    """Tune RF hyperparameters using Randomized Search with k-fold CV.

    Randomly samples *n_iter* combinations from *param_distributions* and
    evaluates each via k-fold CV.  Significantly faster than exhaustive
    Grid Search when the parameter space is large, and typically finds
    near-optimal parameters within fewer iterations.

    Args:
        X_train:             Scaled training feature matrix.
        y_train:             Log1p-transformed training target.
        param_distributions: Mapping of param names → distributions or lists.
                             Defaults to ``PARAM_DISTRIBUTIONS`` from config.
        n_iter:              Number of random combinations to try.
        cv:                  CV folds per combination.
        scoring:             Evaluation metric.
        random_state:        Seed for reproducibility.

    Returns:
        Fitted :class:`~sklearn.model_selection.RandomizedSearchCV`.
        Access best params via ``search.best_params_`` and the best
        fitted model via ``search.best_estimator_``.
    """
    param_distributions = param_distributions or PARAM_DISTRIBUTIONS

    logger.info(
        "RandomizedSearchCV – n_iter=%d, cv=%d, scoring='%s'",
        n_iter, cv, scoring,
    )

    rf = RandomForestRegressor(random_state=random_state)
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
        return_train_score=True,
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 2)

    logger.info(
        "RandomizedSearchCV done in %.1fs — Best %s: %.4f",
        elapsed, scoring.upper(), search.best_score_,
    )
    logger.info("Best params: %s", search.best_params_)
    _save_tuning_results(search, method="randomized_search", elapsed=elapsed)
    return search


def tune_with_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict | None = None,
    cv: int = CV_FOLDS,
    scoring: str = "r2",
    random_state: int = RANDOM_STATE,
) -> GridSearchCV:
    """Tune RF hyperparameters via exhaustive Grid Search.

    Evaluates *every* combination in *param_grid* with k-fold CV.  Use this
    as a secondary fine-tuning step on a small, focused grid after Randomized
    Search has narrowed the parameter space.

    Args:
        X_train:      Scaled training feature matrix.
        y_train:      Log1p-transformed training target.
        param_grid:   Dict of param names → value lists.
                      Defaults to ``FINE_TUNE_GRID`` from config.
        cv:           CV folds per combination.
        scoring:      Evaluation metric.
        random_state: Seed for the RF estimator.

    Returns:
        Fitted :class:`~sklearn.model_selection.GridSearchCV`.
    """
    param_grid = param_grid or FINE_TUNE_GRID

    logger.info(
        "GridSearchCV – cv=%d, scoring='%s', grid size=%d combinations",
        cv, scoring,
        int(
            len(param_grid.get("n_estimators", [1]))
            * len(param_grid.get("max_depth", [1]))
            * len(param_grid.get("min_samples_split", [1]))
            * len(param_grid.get("min_samples_leaf", [1]))
        ),
    )

    rf = RandomForestRegressor(random_state=random_state)
    search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 2)

    logger.info(
        "GridSearchCV done in %.1fs — Best %s: %.4f",
        elapsed, scoring.upper(), search.best_score_,
    )
    logger.info("Best params: %s", search.best_params_)
    _save_tuning_results(search, method="grid_search", elapsed=elapsed)
    return search


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save_cv_results(result: dict) -> None:
    """Persist cross-validation fold scores to ``reports/cv_results.txt``."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "cv_results.txt"

    with open(out, "w", encoding="utf-8") as f:
        f.write("=" * 55 + "\n")
        f.write("RANDOM FOREST — CROSS-VALIDATION RESULTS\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"  Folds   : {result['cv_folds']}\n")
        f.write(f"  Metric  : {result['scoring'].upper()}\n")
        f.write(f"  Elapsed : {result['elapsed_s']}s\n\n")
        f.write("  Per-Fold Scores:\n")
        for i, s in enumerate(result["cv_scores"], start=1):
            f.write(f"    Fold {i}  : {s:.6f}\n")
        f.write(f"\n  Mean    : {result['mean']:.6f}\n")
        f.write(f"  Std Dev : {result['std']:.6f}\n\n")
        f.write("Note: Scores in log-space (log1p target). Notebook R² ≈ 0.968.\n")

    logger.info("CV results saved → %s", out)


def _save_tuning_results(search, method: str, elapsed: float) -> None:
    """Persist best hyperparameter search results to ``reports/tuning_results.txt``."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "tuning_results.txt"

    with open(out, "w", encoding="utf-8") as f:
        f.write("=" * 55 + "\n")
        f.write(f"RANDOM FOREST — HYPERPARAMETER TUNING ({method.upper()})\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"  Method       : {method}\n")
        f.write(f"  Best CV R²   : {search.best_score_:.6f}\n")
        f.write(f"  Elapsed      : {elapsed:.1f}s\n\n")
        f.write("  Best Parameters:\n")
        for param, val in sorted(search.best_params_.items()):
            f.write(f"    {param:<24}: {val}\n")

    logger.info("Tuning results saved → %s", out)
