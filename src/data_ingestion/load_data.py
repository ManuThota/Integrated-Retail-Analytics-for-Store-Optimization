"""
Data Ingestion Module
======================
Loads the three raw CSV datasets:
  - sales.csv     : Weekly store-department sales data
  - stores.csv    : Store metadata (Type, Size)
  - features.csv  : Macroeconomic and promotional features

All file paths are resolved from ``src/config/config.py``,
so the loader is portable across environments.
"""

import logging

import pandas as pd

from src.config.config import SALES_CSV, STORES_CSV, FEATURES_CSV

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Main Flow
# ─────────────────────────────────────────────────────────────────────────────

def load_sales(path=None) -> pd.DataFrame:
    """Load the raw weekly-sales dataset.

    Args:
        path: Optional explicit file path. Defaults to ``data/raw/sales.csv``.

    Returns:
        DataFrame with columns: Store, Dept, Date, Weekly_Sales, IsHoliday.
    """
    file_path = path or SALES_CSV
    logger.info("Loading sales data from: %s", file_path)
    df = pd.read_csv(file_path)
    logger.info("Sales data loaded – shape: %s", df.shape)
    return df


def load_stores(path=None) -> pd.DataFrame:
    """Load the store-metadata dataset.

    Args:
        path: Optional explicit file path. Defaults to ``data/raw/stores.csv``.

    Returns:
        DataFrame with columns: Store, Type, Size.
    """
    file_path = path or STORES_CSV
    logger.info("Loading stores data from: %s", file_path)
    df = pd.read_csv(file_path)
    logger.info("Stores data loaded – shape: %s", df.shape)
    return df


def load_features(path=None) -> pd.DataFrame:
    """Load the economic / promotional features dataset.

    Args:
        path: Optional explicit file path. Defaults to ``data/raw/features.csv``.

    Returns:
        DataFrame with columns: Store, Date, Temperature, Fuel_Price,
        MarkDown1-5, CPI, Unemployment, IsHoliday.
    """
    file_path = path or FEATURES_CSV
    logger.info("Loading features data from: %s", file_path)
    df = pd.read_csv(file_path)
    logger.info("Features data loaded – shape: %s", df.shape)
    return df


def load_all_raw() -> tuple:
    """Convenience wrapper – load all three raw datasets in one call.

    Returns:
        Tuple of (sales_df, stores_df, features_df).
    """
    return load_sales(), load_stores(), load_features()
