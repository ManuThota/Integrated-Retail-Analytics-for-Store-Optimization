"""
feature_builder.py

Handles feature creation and dataset preparation:
- Time features (Year, Month, Week)
- ML dataset creation
- Time-series dataset creation
"""

import pandas as pd


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