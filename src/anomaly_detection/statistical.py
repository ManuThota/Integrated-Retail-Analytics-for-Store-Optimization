"""
statistical.py

Handles statistical anomaly detection:
- IQR-based outlier detection
- Optional outlier treatment
"""

import pandas as pd


def detect_outliers_iqr(df: pd.DataFrame, column: str = "Weekly_Sales") -> pd.DataFrame:
    """
    Detects outliers using IQR method and flags them.

    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column to analyze

    Returns:
        pd.DataFrame: Dataframe with anomaly flag
    """

    df = df.copy()

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Create anomaly flag column
    df["Anomaly"] = ((df[column] < lower_bound) | (df[column] > upper_bound)).astype(int)

    anomaly_count = df["Anomaly"].sum()

    print(f"(✓) -> IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"(✓) -> Total anomalies detected: {anomaly_count}")

    return df


def cap_outliers(df: pd.DataFrame, column: str = "Weekly_Sales") -> pd.DataFrame:
    """
    Caps extreme values using IQR bounds.

    This stabilizes the dataset for ML models.

    Returns:
        pd.DataFrame
    """

    df = df.copy()

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    print("(✓) -> Outliers capped using IQR method")

    return df