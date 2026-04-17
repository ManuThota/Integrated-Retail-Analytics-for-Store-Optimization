"""
transformation.py

Handles:
- Date conversion
- Feature engineering (Year, Month, Week)

"""
#=======================
# # Importing Linraries
#=======================
import pandas as pd

#===========================================================================
# Converting Date Column
#===========================================================================
def convert_date(df: pd.DataFrame, column: str = "Date") -> pd.DataFrame:
    """
    Converts Date column to datetime format.
    """
    df[column] = pd.to_datetime(df[column], errors="coerce")
    return df

#================================================================================
# Creating Time Features
#================================================================================
def create_time_features(df: pd.DataFrame, column: str = "Date") -> pd.DataFrame:
    """
    Creates time-based features:
    - Year
    - Month
    - Week
    """

    df["Year"] = df[column].dt.year
    df["Month"] = df[column].dt.month
    df["Week"] = df[column].dt.isocalendar().week.astype(int)

    print("(✓) -> Time features created")

    return df