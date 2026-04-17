"""
Test feature engineering pipeline
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.config.config import INTERIM_DATA_DIR

from src.feature_engineering.encoding import encode_categorical
from src.feature_engineering.feature_builder import (
    create_time_features,
    create_ml_dataset,
    create_time_series_dataset
)
from src.feature_engineering.feature_selection import (
    correlation_analysis,
    get_final_ml_dataset,
    get_final_ts_dataset
)


def test_feature_engineering():

    print("\n==========( Feature Engineering Test Started )==========\n")

    file_path = INTERIM_DATA_DIR / "merged_data.csv"

    if not file_path.exists():
        raise FileNotFoundError("(✕) -> Run preprocessing first")

    df = pd.read_csv(file_path, parse_dates=["Date"])

    print("(✓) -> Data loaded")

    df = encode_categorical(df)
    df = create_time_features(df)

    df_ml = create_ml_dataset(df)
    df_ts = create_time_series_dataset(df)

    correlation_analysis(df_ml)

    df_ml_final = get_final_ml_dataset(df_ml)
    df_ts_final = get_final_ts_dataset(df_ts)

    assert "Weekly_Sales" in df_ml_final.columns, "(✕) -> Target missing in ML dataset"
    assert "Weekly_Sales" in df_ts_final.columns, "(✕) -> Target missing in TS dataset"

    print("\n(✓) -> Feature Engineering Pipeline Passed Successfully\n")
    print("\n==========( Feature Engineering Test Ended )==========\n")


if __name__ == "__main__":
    test_feature_engineering()