"""
merging.py

Handles merging of datasets:
- Sales
- Stores
- Features

Ensures proper relational joins.
"""
#====================
# Importing Libraries
#====================
import pandas as pd


#====================================================================
# Merging Datasets
#====================================================================
def merge_datasets(sales_df, stores_df, features_df) -> pd.DataFrame:
    """
    Merges all datasets on Store and Date.
    """

    # Merge sales with stores
    df = pd.merge(sales_df, stores_df, on="Store", how="left")

    # Merge with external features
    df = pd.merge(df, features_df, on=["Store", "Date"], how="left")

    print(f"🔗 Merged dataset shape: {df.shape}")

    return df