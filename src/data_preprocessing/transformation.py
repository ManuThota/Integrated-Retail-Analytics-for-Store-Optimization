"""
transformation.py

Handles data transformations such as:
- Converting date columns to datetime
- Extracting time-based features
"""
#=====================
# Importing Libraries
#=====================
import pandas as pd

#==============================================================================
# Convert Date column
#==============================================================================
def convert_date(df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    """
    Converts the Date column to datetime format.

    Args:
        df (pd.DataFrame): Input dataframe
        date_column (str): Name of date column

    Returns:
        pd.DataFrame: Updated dataframe
    """

    df = df.copy()

    # Convert to datetime
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    # Check for invalid conversions
    if df[date_column].isnull().sum() > 0:
        print("(✕) -> Warning: Some dates could not be converted and are NaT")

    print("(✓) -> Date column converted to datetime")

    return df