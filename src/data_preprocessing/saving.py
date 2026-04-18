"""
Saving Module
==============
Persists DataFrames at the interim and processed checkpoints.

  - ``save_interim(df)``   → ``data/interim/merged_data.csv``
    Merged but un-transformed data (after merge, before clean/transform).
    Useful for debugging and EDA on raw-merged data.

  - ``save_processed(df)`` → ``data/processed/final_data.csv``
    Fully cleaned and transformed data ready for modelling.
"""

import logging

import pandas as pd

from src.config.config import INTERIM_MERGED_CSV, PROCESSED_CSV

logger = logging.getLogger(__name__)


def save_interim(df: pd.DataFrame) -> None:
    """Persist the merged (pre-transform) DataFrame to ``data/interim/``.

    Args:
        df: Merged DataFrame (before cleaning / transformation).
    """
    INTERIM_MERGED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTERIM_MERGED_CSV, index=False)
    logger.info("Interim data saved → %s", INTERIM_MERGED_CSV)


def save_processed(df: pd.DataFrame) -> None:
    """Persist the fully preprocessed DataFrame to ``data/processed/``.

    Args:
        df: Fully cleaned and transformed DataFrame.
    """
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False)
    logger.info("Processed data saved → %s", PROCESSED_CSV)
