"""
Model Evaluation Module
========================
Computes test-set metrics for the trained Random Forest Regressor and
generates evaluation reports.

Outputs
────────
``reports/metrics.json``               – machine-readable RMSE / MAE / R².
``reports/figures/feature_importance_mdi.png``
                                        – top-15 MDI feature importance plot.
                                          (Full explainability in
                                          ``src/explainability/feature_importance.py``.)

Metric space
─────────────
All metrics are computed in **log-space** because the pipeline applies
``log1p`` to ``Weekly_Sales`` before training.  The notebook baseline:
  RMSE ≈ 0.34  |  MAE ≈ 0.18  |  R² ≈ 0.97
"""

import json
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config.config import METRICS_JSON, REPORTS_DIR
from src.explainability.feature_importance import plot_mdi_importance, save_importance_report

logger = logging.getLogger(__name__)


def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Compute RMSE, MAE, and R² on the held-out test set.

    Args:
        model:  Fitted :class:`~sklearn.ensemble.RandomForestRegressor`.
        X_test: Scaled test feature matrix.
        y_test: Log1p-transformed test target series.

    Returns:
        Dictionary with keys ``rmse``, ``mae``, ``r2`` (all floats).
    """
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    logger.info(
        "Test-set metrics — RMSE: %.4f | MAE: %.4f | R²: %.4f",
        rmse, mae, r2,
    )
    return metrics


def save_metrics(
    metrics: dict,
    model: RandomForestRegressor,
    feature_names: list[str],
) -> None:
    """Persist evaluation metrics and generate the feature-importance plot.

    Writes:
      - ``reports/metrics.json``                     (machine-readable)
      - ``reports/figures/feature_importance_mdi.png`` (MDI bar chart)
      - Appends feature ranking to ``reports/insights.txt``

    Args:
        metrics:       Dict returned by :func:`evaluate_model`.
        model:         Fitted Random Forest.
        feature_names: Feature column names in the same order as X_train.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Machine-readable metrics ──────────────────────────────────────────────
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "rmse": round(metrics["rmse"], 6),
                "mae":  round(metrics["mae"],  6),
                "r2":   round(metrics["r2"],   6),
                "note": "Metrics in log-space (log1p target). "
                        "Baseline: RMSE≈0.34, MAE≈0.18, R²≈0.97",
            },
            f,
            indent=4,
        )
    logger.info("Model metrics JSON saved → %s", METRICS_JSON)

    # ── Feature importance (MDI) plot + ranking in insights.txt ──────────────
    importances = plot_mdi_importance(model, feature_names)
    save_importance_report(importances)

    logger.info("All evaluation artefacts written to: %s", REPORTS_DIR)
