"""
train.py

Handles:
- Train-test split
- Model training (Random Forest)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from src.config.config import RANDOM_STATE, TEST_SIZE


def split_data(df: pd.DataFrame, target_column: str = "Weekly_Sales"):
    """
    Splits dataset into training and testing sets.

    Returns:
        X_train, X_test, y_train, y_test
    """

    X = df.drop(columns=['Weekly_Sales', 'Weekly_Sales_Log'])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"(✓) -> Data split completed | Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    """
    Trains Random Forest Regressor.

    Returns:
        trained model
    """

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("(✓) -> Random Forest model trained successfully")

    return model