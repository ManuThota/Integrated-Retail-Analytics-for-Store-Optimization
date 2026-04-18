"""
Encoding Module
================
Handles categorical-variable encoding for the retail analytics pipeline.

One-Hot Encoding (OHE) strategy
─────────────────────────────────
The notebook used ``pd.get_dummies`` to convert the store ``Type`` column
(categorical: A / B / C) into binary dummy variables before model training.
This is **intentionally different** from Label Encoding (A→0, B→1, C→2)
which was previously used in this project.

Why OHE instead of Label Encoding?
  - Label encoding implies an ordinal relationship (B > A, C > B) which does
    not exist in store types.
  - OHE treats each type as an independent binary indicator, which is both
    correct and interpretable by the Random Forest.

With ``drop_first=True`` (default):
  - Reference category: Type_A  (dropped to avoid perfect multicollinearity).
  - New columns created: ``Type_B``, ``Type_C``.

Boolean encoding
─────────────────
The ``IsHoliday`` column arrives as Python ``bool`` (True / False).
Scikit-learn estimators accept booleans, but converting to ``int`` (0 / 1)
ensures compatibility with all downstream tools (e.g., scalers, SHAP, CSV
serialisation without losing precision).
"""

import logging

import pandas as pd

from src.config.config import OHE_COLUMNS

logger = logging.getLogger(__name__)


def apply_one_hot_encoding(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    """Convert categorical columns to one-hot dummy variables.

    This function mirrors the ``pd.get_dummies`` call in the notebook.
    The resulting dummy columns are appended to the DataFrame and the
    original categorical column is removed.

    Args:
        df:         Input DataFrame (post-transformation, pre-modelling).
        columns:    List of column names to encode.
                    Defaults to ``OHE_COLUMNS`` from config (``['Type']``).
        drop_first: If True, drop the first dummy column per categorical
                    variable to avoid the dummy-variable trap (default True).
                    With Type having values A / B / C this yields
                    ``Type_B`` and ``Type_C``.

    Returns:
        Copy of *df* with original categorical columns replaced by dummy
        binary columns.

    Example:
        >>> df_encoded = apply_one_hot_encoding(df)
        >>> # df_encoded now has 'Type_B', 'Type_C' instead of 'Type'
    """
    columns = columns or OHE_COLUMNS
    present = [c for c in columns if c in df.columns]
    missing = set(columns) - set(present)

    if missing:
        logger.warning(
            "OHE columns not found in DataFrame and will be skipped: %s", missing
        )

    if not present:
        logger.warning("No OHE columns to encode; returning DataFrame unchanged.")
        return df.copy()

    logger.info(
        "Applying One-Hot Encoding to columns: %s (drop_first=%s) ...",
        present, drop_first,
    )
    df_encoded = pd.get_dummies(df, columns=present, drop_first=drop_first)

    # Log the newly created dummy column names for traceability
    new_cols = [c for c in df_encoded.columns if c not in df.columns]
    logger.info("OHE produced %d new columns: %s", len(new_cols), new_cols)

    return df_encoded


def encode_boolean_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Convert boolean columns to integer (0 / 1) representation.

    Args:
        df:      Input DataFrame.
        columns: Boolean column names to convert.
                 Defaults to ``['IsHoliday']``.

    Returns:
        Copy of *df* with specified columns cast to ``int``.
    """
    columns = columns or ["IsHoliday"]
    df = df.copy()

    for col in columns:
        if col in df.columns and df[col].dtype == bool:
            df[col] = df[col].astype(int)
            logger.info("Converted boolean column '%s' → int (0/1).", col)

    return df
