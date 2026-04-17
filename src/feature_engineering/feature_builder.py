"""
feature_builder.py

Handles feature creation and dataset preparation:
- Time features (Year, Month, Week)
- ML dataset creation
- Time-series dataset creation
"""

import pandas as pd
import numpy as np


def handle_negative_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows where Weekly_Sales is negative.

    This ensures:
    - Valid log transformation
    - Clean target distribution
    """

    df = df.copy()

    negative_count = (df["Weekly_Sales"] < 0).sum()

    print(f"(✓) -> Negative sales detected: {negative_count}")

    df = df[df["Weekly_Sales"] >= 0]

    print(f"(✓) -> Negative sales removed | New shape: {df.shape}")

    return df


def create_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates log-transformed target variable.

    Adds:
    - Weekly_Sales_Log
    """

    df = df.copy()

    df["Weekly_Sales_Log"] = np.log1p(df["Weekly_Sales"])

    print("(✓) -> Log transformation applied on target")

    return df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates time-based features from Date column.
    """

    df = df.copy()

    # Week (important for modeling)
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

    # Year and Month
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    print("(✓) -> Time-based feature creation completed")

    return df


def create_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates dataset for machine learning models.
    Drops Date column.
    """

    df_ml = df.drop("Date", axis=1)

    print("(✓) -> ML dataset created")

    return df_ml


def create_time_series_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates dataset for time-series models.
    """

    df_ts = df.copy()

    # Sort (VERY IMPORTANT)
    df_ts = df_ts.sort_values(by=["Store", "Dept", "Date"])

    # Set index
    df_ts.set_index("Date", inplace=True)

    print("(✓) -> Time-series dataset created")

    return df_ts