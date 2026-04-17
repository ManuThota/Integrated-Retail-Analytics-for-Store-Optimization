"""
Test anomaly detection pipeline
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.config.config import INTERIM_DATA_DIR

from src.anomaly_detection.statistical import detect_outliers_iqr, cap_outliers
from src.anomaly_detection.time_series import detect_time_anomalies


def test_anomaly_pipeline():

    file_path = INTERIM_DATA_DIR / "merged_data.csv"

    if not file_path.exists():
        raise FileNotFoundError("(✕) -> Run preprocessing first")

    df = pd.read_csv(file_path, parse_dates=["Date"])

    print("(✓) -> Loaded data")

    df = detect_outliers_iqr(df)
    df = detect_time_anomalies(df)
    df = cap_outliers(df)

    print("(✓) -> Anomaly pipeline executed successfully")


if __name__ == "__main__":
    test_anomaly_pipeline()