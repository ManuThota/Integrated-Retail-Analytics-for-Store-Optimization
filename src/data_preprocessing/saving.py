"""
saving.py

Handles saving of datasets at different stages:
- Interim (after merging & basic cleaning)
- Processed (after full preprocessing)
"""
#==============================================
# Importing Libraries
#==============================================
import pandas as pd
from pathlib import Path
from src.config.config import INTERIM_DATA_DIR

#===================================================================================
# Saving the Merging Data
#===================================================================================
def save_interim_data(df: pd.DataFrame, filename: str = "merged_data.csv") -> None:
    """
    Saves interim dataset after merging and initial cleaning.

    Args:
        df (pd.DataFrame): Data to save
        filename (str): File name
    """

    output_path = INTERIM_DATA_DIR / filename

    # Ensure directory exists
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"(✓) -> Interim data saved at: {output_path}")