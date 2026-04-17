"""
cleaning.py

Handles:
- Missing values
- Duplicate removal
- Outlier treatment

This ensures clean and reliable data for downstream tasks.

"""
#========================
# Importing Libraries
#========================
import pandas as pd
import numpy as np

#============================================================
# Handles Missing Values
#============================================================
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values:
    - Numerical → median
    - Categorical → 'Unknown'
    """

    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna("Unknown", inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    return df

#============================================================
# Removes Duplicate values
#============================================================
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows.
    """
    initial_shape = df.shape
    df = df.drop_duplicates()
    print(f"🧹 Removed duplicates: {initial_shape[0] - df.shape[0]}")
    return df


#=====================================================================
# Handles Outliers
#=====================================================================
def treat_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Treats outliers using IQR method (capping).
    """

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[column] = np.clip(df[column], lower, upper)

    print(f"(✓) -> Outliers treated for: {column}")

    return df