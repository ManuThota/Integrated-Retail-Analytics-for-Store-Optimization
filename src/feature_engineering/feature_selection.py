"""
Feature Selection Module
=========================
Selects the final feature matrix (X) and target vector (y) from the
fully engineered DataFrame, and performs a train / test split.

Design decisions
─────────────────
* The base feature list (``FEATURE_COLS``) is defined in ``config.py``.
  It deliberately excludes the raw ``Type`` column because that column
  is transformed by One-Hot Encoding into dummy columns (``Type_B``,
  ``Type_C``) in ``encoding.py``.  This function detects those OHE-
  generated columns automatically and appends them to the selection.

* Any column in ``FEATURE_COLS`` that is missing from the DataFrame is
  safely skipped with a warning (e.g., if markdown-aggregate features
  are not yet built).

* Columns that contain ``NaN`` values after the cleaning / encoding
  steps are flagged so that downstream model training does not silently
  receive missing data.
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.config import FEATURE_COLS, TARGET_COL, TEST_SIZE, RANDOM_STATE

logger = logging.getLogger(__name__)


def select_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    base_feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the feature matrix X and target vector y.

    Automatically detects One-Hot Encoded columns (e.g., ``Type_B``,
    ``Type_C``) and appends them to the selection.

    Args:
        df:                Input fully-engineered DataFrame.
        target_col:        Name of the target column (default ``Weekly_Sales``).
        base_feature_cols: Override the base feature list from config.
                           Defaults to ``FEATURE_COLS``.

    Returns:
        Tuple ``(X, y)`` where:
          - X is a DataFrame of selected features.
          - y is a Series of target values (log1p-transformed Weekly_Sales).
    """
    base_cols = base_feature_cols or FEATURE_COLS

    # Discover OHE-generated columns (e.g. Type_B, Type_C from pd.get_dummies)
    ohe_cols = [c for c in df.columns if c.startswith("Type_")]

    # Combine base features + OHE columns, filtering to only existing columns
    wanted    = [c for c in base_cols if c != "Type"] + ohe_cols
    available = [c for c in wanted if c in df.columns]
    skipped   = set(wanted) - set(available)

    if skipped:
        logger.warning(
            "The following feature columns are missing from the DataFrame "
            "and will be excluded from X: %s",
            sorted(skipped),
        )

    X = df[available].copy()
    y = df[target_col].copy()

    # Warn about NaN values that could silently corrupt model training
    nan_counts = X.isnull().sum()
    nan_cols   = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        logger.warning("NaN values found in feature columns:\n%s", nan_cols)

    logger.info(
        "Feature matrix X: shape %s | Target y: %d values",
        X.shape, len(y),
    )
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a random train / test split.

    The notebook uses random (non-temporal) splitting for the ML models
    because Random Forest does not rely on temporal ordering.

    Args:
        X:            Feature matrix.
        y:            Target series.
        test_size:    Fraction held out for testing (default from config: 0.2).
        random_state: Reproducibility seed (default from config: 42).

    Returns:
        Tuple ``(X_train, X_test, y_train, y_test)``.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Data split – Train: %d rows (%.0f%%)  |  Test: %d rows (%.0f%%)",
        len(X_train), (1 - test_size) * 100,
        len(X_test),  test_size * 100,
    )
    return X_train, X_test, y_train, y_test
