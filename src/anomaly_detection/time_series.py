"""
time_series.py

Handles time-based anomaly detection:
- Weekly trend deviations
- Rolling mean comparison
"""

import pandas as pd


def detect_time_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects anomalies based on time trends using rolling statistics.

    Returns:
        pd.DataFrame
    """

    df = df.copy()

    # Sort by date
    df = df.sort_values("Date")

    # Rolling mean and std (window = 4 weeks)
    df["Rolling_Mean"] = df["Weekly_Sales"].rolling(window=4).mean()
    df["Rolling_Std"] = df["Weekly_Sales"].rolling(window=4).std()

    # Define anomaly condition
    df["Time_Anomaly"] = (
        abs(df["Weekly_Sales"] - df["Rolling_Mean"]) > (2 * df["Rolling_Std"])
    ).astype(int)

    anomaly_count = df["Time_Anomaly"].sum()

    print(f"(✓) -> Time-based anomalies detected: {anomaly_count}")

    return df