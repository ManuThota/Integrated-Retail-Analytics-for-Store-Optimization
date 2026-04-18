"""
Scaler Module
==============
Wraps scikit-learn's ``StandardScaler`` with project-aware serialisation.

Why StandardScaler?
  The Random Forest algorithm itself is scale-invariant (splits are based on
  feature thresholds, not distances).  However, scaling is applied here for
  consistency with the notebook pipeline and to ensure compatibility with
  any downstream models or analysis tools that *are* scale-sensitive
  (e.g., permutation importance comparisons, PCA).

  Critically, the scaler is **fit only on the training split** and then
  applied to the test split — this prevents data leakage from the test set
  into the normalisation parameters.

Outputs
────────
``models/scaler.pkl``  ← joblib-serialised fitted scaler (path from config).
"""

import logging

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config.config import SCALER_PATH

logger = logging.getLogger(__name__)


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a new ``StandardScaler`` on the training feature matrix.

    Args:
        X_train: Training feature matrix (pre-scaling).

    Returns:
        Fitted :class:`sklearn.preprocessing.StandardScaler`.
    """
    logger.info("Fitting StandardScaler on training data (shape: %s) ...", X_train.shape)
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_data(
    X: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Apply a pre-fitted scaler to a feature matrix.

    Returns a new DataFrame (preserving column names) rather than a raw
    numpy array, ensuring downstream code that relies on column names
    (e.g., feature importance lookups) remains functional.

    Args:
        X:      Feature matrix to scale.
        scaler: A **previously fitted** ``StandardScaler``.

    Returns:
        Scaled DataFrame with the same columns and index as *X*.
    """
    scaled_array = scaler.transform(X)
    return pd.DataFrame(scaled_array, columns=X.columns, index=X.index)


def save_scaler(
    scaler: StandardScaler,
    path=None,
) -> None:
    """Serialise the fitted scaler to disk using joblib.

    Args:
        scaler: Fitted ``StandardScaler`` to persist.
        path:   Output path.  Defaults to ``SCALER_PATH`` from config.
    """
    output_path = path or SCALER_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_path)
    logger.info("Scaler saved → %s", output_path)


def load_scaler(path=None) -> StandardScaler:
    """Load a previously persisted scaler from disk.

    Args:
        path: File path.  Defaults to ``SCALER_PATH`` from config.

    Returns:
        Loaded :class:`sklearn.preprocessing.StandardScaler`.
    """
    file_path = path or SCALER_PATH
    logger.info("Loading scaler from: %s", file_path)
    return joblib.load(file_path)
