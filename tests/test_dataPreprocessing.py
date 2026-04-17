"""
test_dataPreprocessing.py

This script tests the full data preprocessing pipeline using real data.

It validates:
- Data loading
- Date conversion
- Dataset merging
- Missing value handling
- Holiday column fix
- Interim data saving

Run this script before training models to ensure pipeline correctness.
"""

# ============================================================
# Imports
# ============================================================

import sys
import traceback

# Add project root to path (important for imports)
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ingestion.load_data import load_all_data
from src.data_preprocessing.transformation import convert_date
from src.data_preprocessing.merging import merge_datasets
from src.data_preprocessing.cleaning import (
    handle_missing_values,
    fix_holiday_columns
)
from src.data_preprocessing.saving import save_interim_data


# ============================================================
# Test Function
# ============================================================

def test_data_preprocessing_pipeline():
    """
    Runs full preprocessing pipeline and validates each step.
    """

    try:
        print("\nStarting Data Preprocessing Test...\n")

        # ----------------------------------------------------
        # Step 1: Load Data
        # ----------------------------------------------------
        sales_df, stores_df, features_df = load_all_data()

        assert sales_df is not None, "Sales data not loaded"
        assert stores_df is not None, "Stores data not loaded"
        assert features_df is not None, "Features data not loaded"

        print("(✓) -> Data loading test passed\n")

        # ----------------------------------------------------
        # Step 2: Date Conversion
        # ----------------------------------------------------
        sales_df = convert_date(sales_df)
        features_df = convert_date(features_df)

        assert "Date" in sales_df.columns, "(✕) -> Date column missing in sales"
        assert "Date" in features_df.columns, "(✕) -> Date column missing in features"

        print("Date conversion test passed\n")

        # ----------------------------------------------------
        # Step 3: Merge
        # ----------------------------------------------------
        merged_df = merge_datasets(sales_df, stores_df, features_df)

        assert merged_df.shape[0] > 0, "Merge resulted in empty dataframe"

        print("(✓) -> Merging test passed\n")

        # ----------------------------------------------------
        # Step 4: Missing Value Handling
        # ----------------------------------------------------
        cleaned_df = handle_missing_values(merged_df)

        markdown_cols = [col for col in cleaned_df.columns if "MarkDown" in col]

        # Ensure no missing values in MarkDown columns
        for col in markdown_cols:
            assert cleaned_df[col].isnull().sum() == 0, f"Missing values still present in {col}"

        print("(✓) -> Missing value handling test passed\n")

        # ----------------------------------------------------
        # Step 5: Fix Holiday Columns
        # ----------------------------------------------------
        final_df = fix_holiday_columns(cleaned_df)

        assert "IsHoliday" in final_df.columns, "(✕) -> Final IsHoliday column missing"
        assert "IsHoliday_x" not in final_df.columns, "(✕) -> IsHoliday_x not removed"
        assert "IsHoliday_y" not in final_df.columns, "(✕) -> IsHoliday_y not removed"

        print("(✓) -> Holiday column fix test passed\n")

        # ----------------------------------------------------
        # Step 6: Save Interim Data
        # ----------------------------------------------------
        save_interim_data(final_df)

        print("(✓) -> Interim data saving test passed\n")

        # ----------------------------------------------------
        # Final Success
        # ----------------------------------------------------
        print("(✓) -> ALL DATA PREPROCESSING TESTS PASSED SUCCESSFULLY!\n")

    except Exception as e:
        print("\nTEST FAILED!\n")
        print("Error:", str(e))
        print("\nTraceback:")
        traceback.print_exc()


# ============================================================
# Run Test
# ============================================================

if __name__ == "__main__":
    test_data_preprocessing_pipeline()