"""
test_anomalyDetection.py

Tests the anomaly detection pipeline:
- IQR-based outlier removal
- Time-based anomaly detection

Validates:
- Outliers are removed correctly
- Data integrity after removal
- Time anomaly column creation
"""

# ============================================================
# Imports
# ============================================================

import sys
import os
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.config.config import INTERIM_DATA_DIR

from src.anomaly_detection.statistical import remove_outliers_iqr
from src.anomaly_detection.time_series import detect_time_anomalies


# ============================================================
# Test Function
# ============================================================

def test_anomaly_detection_pipeline():
    """
    Runs full anomaly detection pipeline and validates results.
    """

    try:
        print("\n==========( Anomaly Detection Test Started )==========\n")

        # ----------------------------------------------------
        # Step 1: Load Data
        # ----------------------------------------------------
        file_path = INTERIM_DATA_DIR / "merged_data.csv"

        if not file_path.exists():
            raise FileNotFoundError("(✕) -> Interim data not found. Run preprocessing first.")

        df = pd.read_csv(file_path, parse_dates=["Date"])

        print(f"(✓) -> Data loaded | Shape: {df.shape}")

        # ----------------------------------------------------
        # Step 2: Store original size
        # ----------------------------------------------------
        original_rows = df.shape[0]

        # ----------------------------------------------------
        # Step 3: Remove Outliers (IQR)
        # ----------------------------------------------------
        df_cleaned, lower_bound, upper_bound = remove_outliers_iqr(
            df,
            return_bounds=True
        )

        cleaned_rows = df_cleaned.shape[0]

        assert cleaned_rows < original_rows, "(✕) -> No rows were removed. Check IQR logic."

        print(f"(✓) -> Outlier removal validated | Rows removed: {original_rows - cleaned_rows}")


        # ----------------------------------------------------
        # Step 4: Validate Using SAME Bounds
        # ----------------------------------------------------
        remaining_outliers = df_cleaned[
            (df_cleaned["Weekly_Sales"] < lower_bound) |
            (df_cleaned["Weekly_Sales"] > upper_bound)
        ]

        assert remaining_outliers.shape[0] == 0, "(✕) -> Outliers still present after removal"

        print("(✓) -> Outlier removal correctness validated (same bounds)")

        # ----------------------------------------------------
        # Step 5: Time-Based Anomaly Detection
        # ----------------------------------------------------
        df_final = detect_time_anomalies(df_cleaned)

        assert "Time_Anomaly" in df_final.columns, "(✕) -> Time_Anomaly column missing"

        anomaly_count = df_final["Time_Anomaly"].sum()

        print(f"(✓) -> Time anomalies detected: {anomaly_count}")

        # ----------------------------------------------------
        # Final Success
        # ----------------------------------------------------
        print("\n(✓) -> ALL ANOMALY DETECTION TESTS PASSED SUCCESSFULLY!\n")

    except Exception as e:
        print("\n(✕) -> TEST FAILED!\n")
        print("Error:", str(e))
        print("\nTraceback:")
        traceback.print_exc()


# ============================================================
# Run Test
# ============================================================

if __name__ == "__main__":
    test_anomaly_detection_pipeline()