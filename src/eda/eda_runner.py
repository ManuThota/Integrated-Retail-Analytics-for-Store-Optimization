"""
run_eda.py

Runs full EDA pipeline:
- Generates plots
- Saves them automatically
"""

from src.data_ingestion.load_data import load_all_data
from src.data_preprocessing.transformation import convert_date
from src.data_preprocessing.merging import merge_datasets
from src.data_preprocessing.cleaning import handle_missing_values, fix_holiday_columns

from src.eda.univariate import sales_distribution
from src.eda.bivariate import sales_by_store, holiday_impact
from src.eda.multivariate import monthly_trend
from src.eda.visualization import save_plot


def run_eda_pipeline():
    print("\n==========( EDA Pipeline Started )==========\n")

    # Load and preprocess data
    sales, stores, features = load_all_data()

    sales = convert_date(sales)
    features = convert_date(features)

    df = merge_datasets(sales, stores, features)
    df = handle_missing_values(df)
    df = fix_holiday_columns(df)

    # -----------------------------
    # Univariate
    # -----------------------------
    plot = sales_distribution(df)
    save_plot("sales_distribution")

    # -----------------------------
    # Bivariate
    # -----------------------------
    plot = sales_by_store(df)
    save_plot("sales_by_store")

    plot = holiday_impact(df)
    save_plot("holiday_impact")

    # -----------------------------
    # Multivariate
    # -----------------------------
    plot = monthly_trend(df)
    save_plot("monthly_trend")

    print("\n(✓) -> EDA Pipeline Completed Successfully\n")
    print("\n==========( EDA Pipeline Ended )==========\n")


if __name__ == "__main__":
    run_eda_pipeline()