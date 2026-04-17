"""
encoding.py

Handles categorical encoding:
- One-hot encoding for store type
"""

import pandas as pd


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies one-hot encoding to categorical columns.

    Currently encodes:
    - Store Type

    Returns:
        pd.DataFrame
    """

    df = df.copy()

    # One-hot encode 'Type'
    df_encoded = pd.get_dummies(df, columns=["Type"], drop_first=True)

    print("(✓) -> Categorical encoding completed (Type)")

    return df_encoded