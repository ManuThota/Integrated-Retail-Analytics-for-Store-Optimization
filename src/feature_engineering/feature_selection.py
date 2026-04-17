"""
feature_selection.py

Handles:
- Correlation analysis
- Feature selection
"""

import pandas as pd


def correlation_analysis(df_ml: pd.DataFrame) -> pd.Series:
    """
    Computes correlation with target variable.
    """

    corr_matrix = df_ml.corr(numeric_only=True)
    corr_target = corr_matrix["Weekly_Sales"].sort_values(ascending=False)

    print("(✓) -> Correlation analysis completed")
    print(corr_target)

    return corr_target


def get_final_ml_dataset(df_ml: pd.DataFrame) -> pd.DataFrame:
    """
    Returns final ML dataset.
    (Currently using all features as per notebook)
    """

    df_final = df_ml.copy()

    print("(✓) -> Final ML dataset ready")

    return df_final


def get_final_ts_dataset(df_ts: pd.DataFrame) -> pd.DataFrame:
    """
    Selects relevant features for time-series modeling.
    """

    ts_features = [
        "Store", "Dept", "Weekly_Sales",
        "Temperature", "Fuel_Price",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
        "CPI", "Unemployment",
        "IsHoliday", "Size", "Type_B", "Type_C"
    ]

    df_ts_final = df_ts[ts_features]

    print("(✓) -> Final time-series dataset ready")

    return df_ts_final