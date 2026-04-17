"""
load_data.py

This module handles all data ingestion tasks:
- Loading raw datasets
- Validating file existence
- Basic sanity checks

"""

#========================
# Importing Libraries
#========================
import pandas as pd
from pathlib import Path


#==============================
# Import paths from config
#==============================

from src.config.config import (
    SALES_DATA_PATH,
    STORES_DATA_PATH,
    FEATURES_DATA_PATH
)

#================================================================
# Checking the files existance
#================================================================

def check_file_exists(file_path: Path) -> None:
    """
    Checks whether a given file exists.

    Args:
        file_path (Path): Path to the file

    Raises:
        FileNotFoundError: If file does not exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"(✕) -> File not found: {file_path}")


#======================================================================
# Loading the Datasets into Pandas Dataframe
#======================================================================
def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file_path (Path): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        print(f"(✓) -> Loaded: {file_path.name} | Shape: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"(▲) -> Error loading {file_path.name}: {e}")

#===========================================================
# Validating the Dataframe
#===========================================================
def validate_dataframe(df: pd.DataFrame, name: str) -> None:
    """
    Performs basic validation on the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to validate
        name (str): Name of dataset
    """
    if df.empty:
        raise ValueError(f"(▲) -> {name} is empty!")

    print(f"(✓) -> {name} validation passed")
    print(f"   Columns: {list(df.columns)}\n")


#==================================================
# Main Function to load the datasets
#==================================================
def load_all_data():
    """
    Main function to load all datasets.

    Returns:
        tuple: (sales_df, stores_df, features_df)
    """

    print("==========( Data Ingestion Started )==========")

    # Step 1: Check file existence
    check_file_exists(SALES_DATA_PATH)
    check_file_exists(STORES_DATA_PATH)
    check_file_exists(FEATURES_DATA_PATH)
    print("----------------------------------------------")
    # Step 2: Load datasets
    sales_df = load_csv(SALES_DATA_PATH)
    stores_df = load_csv(STORES_DATA_PATH)
    features_df = load_csv(FEATURES_DATA_PATH)
    print("----------------------------------------------")
    # Step 3: Validate datasets
    validate_dataframe(sales_df, "Sales Data")
    validate_dataframe(stores_df, "Stores Data")
    validate_dataframe(features_df, "Features Data")

    print("==========( Data Ingestion Ended )==========")

    return sales_df, stores_df, features_df


# Run independently for testing
if __name__ == "__main__":
    load_all_data()