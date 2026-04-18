"""
Merging Module
===============
Joins the three raw DataFrames into one unified dataset.

Merge strategy (mirrors the notebook):
  1. sales   LEFT JOIN stores   ON Store
  2. (1)     LEFT JOIN features ON Store + Date + IsHoliday
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def merge_datasets(
    sales_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all three raw DataFrames into a single unified DataFrame.

    Args:
        sales_df:    Raw sales DataFrame.
        stores_df:   Raw stores DataFrame.
        features_df: Raw features DataFrame.

    Returns:
        Merged DataFrame.
    """
    logger.info("Merging sales with stores ...")
    merged = pd.merge(sales_df, stores_df, on="Store", how="left")

    logger.info("Merging with features ...")
    merged = pd.merge(
        merged,
        features_df,
        on=["Store", "Date", "IsHoliday"],
        how="left",
    )

    logger.info("Merge complete – shape: %s", merged.shape)
    return merged
