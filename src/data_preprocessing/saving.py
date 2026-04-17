"""
saving.py

Handles saving datasets at different pipeline stages:

1. Interim Data:
   - After merging datasets
   - Before cleaning

2. Processed Data:
   - After cleaning and preprocessing
   - Ready for feature engineering / modeling
"""

import pandas as pd
from pathlib import Path

from src.config.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR
)


# ============================================================
# Save Interim Data
# ============================================================

def save_interim_data(df: pd.DataFrame, filename: str = "merged_data.csv") -> None:
    """
    Saves merged dataset (before cleaning).

    Args:
        df (pd.DataFrame): Merged dataframe
        filename (str): Output file name
    """

    # Ensure directory exists
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_path = INTERIM_DATA_DIR / filename

    df.to_csv(file_path, index=False)

    print(f"(✓) -> Interim data saved at: {file_path}")


# ============================================================
# Save Processed Data
# ============================================================

def save_processed_data(df: pd.DataFrame, filename: str = "processed_data.csv") -> None:
    """
    Saves cleaned and preprocessed dataset.

    Args:
        df (pd.DataFrame): Cleaned dataframe
        filename (str): Output file name
    """

    # Ensure directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_path = PROCESSED_DATA_DIR / filename

    df.to_csv(file_path, index=False)

    print(f"(✓) -> Processed data saved at: {file_path}")

# ============================================================
# Save Final ML Dataset
# ============================================================

def save_ml_dataset(df: pd.DataFrame, filename: str = "ml_dataset.csv") -> None:
    """
    Saves final machine learning dataset.
    """

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    file_path = PROCESSED_DATA_DIR / filename

    df.to_csv(file_path, index=False)

    print(f"(✓) -> ML dataset saved at: {file_path}")
