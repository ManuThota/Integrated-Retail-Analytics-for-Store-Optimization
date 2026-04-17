"""
saving.py

Handles saving of datasets:
- Interim data (after merging)
- Processed data (after cleaning + feature engineering)

This improves:
- Reproducibility
- Debugging
- Pipeline efficiency
"""
#============================================================
# Importing Libraries
#============================================================
import pandas as pd
from src.config.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR


#===================================================================================
# Saving Interim Data
#===================================================================================
def save_interim_data(df: pd.DataFrame, filename: str = "merged_data.csv") -> None:
    """
    Saves merged (intermediate) dataset.
    """

    path = INTERIM_DATA_DIR / filename
    df.to_csv(path, index=False)

    print(f"(✓) -> Interim data saved at: {path}")

#===================================================================================
# Saving Processed Data
#===================================================================================    
def save_processed_data(df: pd.DataFrame, filename: str = "processed_data.csv") -> None:
    """
    Saves fully processed dataset.
    """

    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=False)

    print(f"(✓) -> Processed data saved at: {path}")