"""
statistical.py

Handles statistical anomaly detection:
- IQR-based outlier detection
- Outlier removal 
"""

import pandas as pd


def remove_outliers_iqr(df: pd.DataFrame,
                       column: str = "Weekly_Sales",
                       return_bounds: bool = False):
    """
    Removes outliers using IQR method.

    Optionally returns bounds for validation.
    """

    df = df.copy()

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)

    cleaned_df = df[mask]

    removed_count = len(df) - len(cleaned_df)

    print(f"(✓) -> IQR bounds calculated: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"(✓) -> Outliers removed: {removed_count}")
    print(f"(✓) -> Remaining data shape: {cleaned_df.shape}")

    if return_bounds:
        return cleaned_df, lower_bound, upper_bound

    return cleaned_df