"""
merging.py

Handles merging of:
- Sales data
- Store data
- External features data
"""
#=====================
# Importing Libraries
#=====================
import pandas as pd

#============================================================
# Merging the Datasets
#============================================================
def merge_datasets(sales_df: pd.DataFrame,
                   stores_df: pd.DataFrame,
                   features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges sales, stores, and features datasets.

    Returns:
        pd.DataFrame: Merged dataframe
    """

    print("Merging datasets...")

    # Merge sales + features on Store & Date
    df = pd.merge(
        sales_df,
        features_df,
        on=["Store", "Date"],
        how="left"
    )

    # Merge with store data
    df = pd.merge(
        df,
        stores_df,
        on="Store",
        how="left"
    )

    print(f"(✓) -> Merging completed | Shape: {df.shape}")

    return df