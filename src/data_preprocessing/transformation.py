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
    df = df.copy()

    # Convert with strict format 
    df[date_column] = pd.to_datetime(
        df[date_column],
        format="%d-%m-%Y",   
        errors="coerce"
    )

    # Drop invalid dates 
    before = len(df)
    df = df.dropna(subset=[date_column])
    after = len(df)

    if before != after:
        print(f"Dropped {before - after} rows due to invalid dates")

    print("Date column converted to datetime")

    return df