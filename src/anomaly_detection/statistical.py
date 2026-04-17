"""
statistical.py

Handles statistical anomaly detection:
- IQR-based outlier detection
- Outlier removal (as per project notebook approach)
"""

import pandas as pd


def remove_outliers_iqr(df: pd.DataFrame, column: str = "Weekly_Sales") -> pd.DataFrame:
    """
    Removes outliers using IQR method.

    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Target column for outlier detection

    Returns:
        pd.DataFrame: Cleaned dataframe (outliers removed)
    """

    df = df.copy()

    # ----------------------------------------------------
    # Step 1: Calculate IQR
    # ----------------------------------------------------
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"(✓) -> IQR bounds calculated: [{lower_bound:.2f}, {upper_bound:.2f}]")

    # ----------------------------------------------------
    # Step 2: Count outliers
    # ----------------------------------------------------
    before_count = df.shape[0]

    df_cleaned = df[
        (df[column] >= lower_bound) &
        (df[column] <= upper_bound)
    ]

    after_count = df_cleaned.shape[0]

    removed = before_count - after_count

    print(f"(✓) -> Outliers removed: {removed}")
    print(f"(✓) -> Remaining data shape: {df_cleaned.shape}")

    return df_cleaned