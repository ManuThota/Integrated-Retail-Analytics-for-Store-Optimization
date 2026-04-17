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
def merge_datasets(sales_df, stores_df, features_df):

    print("Merging datasets...")

    # Debug: check duplicates
    print("\nChecking key uniqueness...")

    sales_duplicates = sales_df.duplicated(subset=["Store", "Date"]).sum()
    features_duplicates = features_df.duplicated(subset=["Store", "Date"]).sum()

    print(f"Sales duplicates (Store, Date): {sales_duplicates}")
    print(f"Features duplicates (Store, Date): {features_duplicates}")

    # Merge sales + features
    df = pd.merge(
        sales_df,
        features_df,
        on=["Store", "Date"],
        how="left",
        validate="many_to_one" 
    )

    # Merge stores
    df = pd.merge(
        df,
        stores_df,
        on="Store",
        how="left",
        validate="many_to_one"
    )

    print(f"(✓) ->Merging completed | Shape: {df.shape}")

    return df