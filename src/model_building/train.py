"""
Random Forest Trainer Module
==============================
Trains the finalised Random Forest Regressor model as identified in the
notebook after evaluating five candidate models:
  - Linear Regression          (baseline)
  - Decision Tree Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor
  - Random Forest Regressor    ← WINNER  (R² ≈ 0.97, selected for production)

Hyperparameters are centralised in ``src/config/config.py``:
  - N_ESTIMATORS = 100   (trees in the forest)
  - RANDOM_STATE  = 42   (reproducibility seed)

The trained model is persisted to ``models/random_forest.pkl``
(path resolved from ``MODEL_PATH`` in config.py).
"""

import logging

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config.config import MODEL_PATH, N_ESTIMATORS, RANDOM_STATE

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
    save_model: bool = True,
) -> RandomForestRegressor:
    """Train the Random Forest Regressor on the scaled training data.

    Uses ``n_jobs=-1`` to parallelise tree construction across all CPU cores,
    significantly reducing training time on the ~420K-row Walmart dataset.

    Args:
        X_train:      Scaled training feature matrix (output of
                      :func:`src.preprocessing.scaler.transform_data`).
        y_train:      Log1p-transformed training target series.
        n_estimators: Number of trees in the forest.
                      Default: ``N_ESTIMATORS`` = 100 (from config.py).
        random_state: Global reproducibility seed.
                      Default: ``RANDOM_STATE`` = 42 (from config.py).
        save_model:   If True, persist the fitted model to
                      ``models/random_forest.pkl``.

    Returns:
        Fitted :class:`sklearn.ensemble.RandomForestRegressor`.
    """
    logger.info(
        "Training Random Forest Regressor "
        "(n_estimators=%d, random_state=%d, n_jobs=-1) ...",
        n_estimators, random_state,
    )
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,           # use all CPU cores for parallel tree building
    )
    rf.fit(X_train, y_train)
    logger.info("Training complete.")

    if save_model:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf, MODEL_PATH)
        logger.info("Model persisted → %s", MODEL_PATH)

    return rf


def load_model(path=None) -> RandomForestRegressor:
    """Load a previously persisted Random Forest model from disk.

    Args:
        path: Explicit file path override.
              Defaults to ``MODEL_PATH`` from config.py
              (``models/random_forest.pkl``).

    Returns:
        Deserialized :class:`sklearn.ensemble.RandomForestRegressor`.
    """
    file_path = path or MODEL_PATH
    logger.info("Loading model from: %s", file_path)
    model = joblib.load(file_path)
    logger.info("Model loaded successfully.")
    return model
