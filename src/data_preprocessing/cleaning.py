"""
cleaning.py

Handles:
- Missing values
- MarkDown handling
- Duplicate column fixes (IsHoliday_x, IsHoliday_y)
"""
#=====================
# Importing Libraries
#=====================
import pandas as pd

#============================================================
# Handling Missing Values
#============================================================
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the dataset.

    Strategy:
    - MarkDown columns → fill with 0 (No promotion assumption)
    """

    df = df.copy()

    print("Missing values before cleaning:\n", df.isnull().sum())

    # Identify MarkDown columns
    markdown_cols = [col for col in df.columns if "MarkDown" in col]

    # Fill MarkDown missing values with 0
    df[markdown_cols] = df[markdown_cols].fillna(0)

    print("(✓) -> Missing values in MarkDown columns handled (filled with 0)")

    return df

#============================================================
# Fixing the Duplicate Holiday column
#============================================================
def fix_holiday_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles duplicate holiday columns after merge.

    Keeps one IsHoliday column and removes duplicates.
    """

    df = df.copy()

    # Check if both columns exist
    if "IsHoliday_x" in df.columns and "IsHoliday_y" in df.columns:

        print("Found duplicate holiday columns. Fixing...")

        # Keep one column (they are usually identical)
        df["IsHoliday"] = df["IsHoliday_x"]

        # Drop duplicates
        df.drop(columns=["IsHoliday_x", "IsHoliday_y"], inplace=True)

        print("Duplicate holiday columns resolved")

    return df