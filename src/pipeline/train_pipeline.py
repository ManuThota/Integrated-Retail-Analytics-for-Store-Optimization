"""
Training Pipeline Module
==========================
Orchestrates Stages 5–8 of the end-to-end pipeline:

  Stage 5 – Feature Engineering : OHE, bool encoding, feature select, split, scale.
  Stage 6 – Cross-Validation    : 5-fold CV to confirm baseline R² ≈ 0.968.
  Stage 7 – Model Training      : Final RF fit + persistence.
  Stage 8 – Evaluation          : Test-set metrics + reports.

Usage
──────
Called by :func:`src.pipeline.full_pipeline.run_pipeline` with the
DataFrame returned from the data pipeline, or standalone::

    from src.pipeline.train_pipeline import run_train_pipeline
    metrics = run_train_pipeline()   # loads from data/processed/final_data.csv
"""

import logging
import time

import pandas as pd

from src.config.config import PROCESSED_CSV
from src.feature_engineering.encoding import apply_one_hot_encoding, encode_boolean_columns
from src.feature_engineering.feature_selection import select_features, split_data
from src.preprocessing.scaler import fit_scaler, transform_data, save_scaler
from src.model_building.tune import run_cross_validation
from src.model_building.train import train_model
from src.model_building.evaluate import evaluate_model, save_metrics
from src.utils.helpers import format_duration

logger = logging.getLogger(__name__)

# ── Terminal banner helpers ───────────────────────────────────────────────────
def _stage(n: int, total: int, msg: str) -> None:
    print(f"\n  [ Stage {n}/{total} ]  ⟳  {msg}")


def _done(msg: str) -> None:
    print(f"               ✓  {msg}")


def run_train_pipeline(
    df: pd.DataFrame | None = None,
    run_cv: bool = True,
) -> dict:
    """Execute the feature engineering, CV, training, and evaluation stages.

    Args:
        df:      Pre-processed DataFrame (output of
                 :func:`src.pipeline.data_pipeline.run_data_pipeline`).
                 If ``None``, loads from ``data/processed/final_data.csv``.
        run_cv:  Whether to run 5-fold cross-validation (default ``True``).

    Returns:
        Dictionary with keys ``cv`` (CV result dict) and ``metrics``
        (test-set evaluation metrics dict).
    """
    # ── Load from disk if not provided ────────────────────────────────────────
    if df is None:
        logger.info("Loading processed data from: %s", PROCESSED_CSV)
        df = pd.read_csv(PROCESSED_CSV)

    # ── Stage 5: Feature Engineering ──────────────────────────────────────────
    _stage(5, 8, "Feature engineering  (OHE → select → split → scale) ...")
    t0 = time.time()

    # 5a. Convert IsHoliday bool → int (0/1)
    df = encode_boolean_columns(df, columns=["IsHoliday"])

    # 5b. One-Hot Encode the 'Type' column  (A/B/C → Type_B, Type_C)
    #     This matches the notebook's pd.get_dummies approach.
    df = apply_one_hot_encoding(df, columns=["Type"], drop_first=True)

    # 5c. Select feature matrix X and target y
    #     OHE dummy columns (Type_B, Type_C) are auto-detected.
    X, y = select_features(df)

    # 5d. Train / test split (80% / 20%, random_state=42)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 5e. Fit StandardScaler on training data ONLY (prevents leakage)
    scaler         = fit_scaler(X_train)
    X_train_scaled = transform_data(X_train, scaler)
    X_test_scaled  = transform_data(X_test,  scaler)
    save_scaler(scaler)

    _done(
        f"Features: {X.shape[1]} columns  |  "
        f"Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  "
        f"Scaler → models/scaler.pkl  ({format_duration(time.time() - t0)})"
    )

    # ── Stage 6: Cross-Validation ─────────────────────────────────────────────
    cv_result = None
    if run_cv:
        _stage(6, 8, "5-fold cross-validation  (this may take a few minutes) ...")
        t0 = time.time()
        cv_result = run_cross_validation(X_train_scaled, y_train)
        _done(
            f"CV R²: {cv_result['mean']:.4f} ± {cv_result['std']:.4f}  "
            f"({format_duration(time.time() - t0)})  → reports/cv_results.txt"
        )
    else:
        _stage(6, 8, "Cross-validation skipped.")
        _done("Skipped (run_cv=False)")

    # ── Stage 7: Final Model Training ─────────────────────────────────────────
    _stage(7, 8, "Training Random Forest Regressor  (n_estimators=100, n_jobs=-1) ...")
    t0 = time.time()
    model = train_model(X_train_scaled, y_train, save_model=True)
    _done(
        f"Model trained → models/random_forest.pkl  "
        f"({format_duration(time.time() - t0)})"
    )

    # ── Stage 8: Evaluation & Reporting ───────────────────────────────────────
    _stage(8, 8, "Evaluating model on hold-out test set ...")
    t0 = time.time()
    metrics = evaluate_model(model, X_test_scaled, y_test)
    save_metrics(metrics, model, feature_names=list(X_train.columns))
    _done(
        f"RMSE: {metrics['rmse']:.4f}  |  "
        f"MAE:  {metrics['mae']:.4f}  |  "
        f"R²:   {metrics['r2']:.4f}  "
        f"({format_duration(time.time() - t0)})"
    )
    _done("metrics.json + feature_importance_mdi.png → reports/")

    return {"cv": cv_result, "metrics": metrics}
