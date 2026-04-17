from pathlib import Path
import pandas as pd

from src.config.config import INTERIM_DATA_DIR

from src.eda.univariate import sales_distribution
from src.eda.bivariate import sales_by_store, holiday_impact
from src.eda.multivariate import monthly_trend
from src.eda.visualization import save_plot


def run_eda_pipeline():
    print("\n==========( EDA Pipeline Started )==========\n")

    # ----------------------------------------------------
    # Load preprocessed (interim) data
    # ----------------------------------------------------
    file_path = INTERIM_DATA_DIR / "merged_data.csv"

    if not file_path.exists():
        raise FileNotFoundError("(✕) -> Interim data not found. Run preprocessing first.")

    df = pd.read_csv(file_path, parse_dates=["Date"])

    print(f"(✓) -> Loaded interim data | Shape: {df.shape}")

    # ----------------------------------------------------
    # Univariate
    # ----------------------------------------------------
    plot = sales_distribution(df)
    save_plot("sales_distribution")

    # ----------------------------------------------------
    # Bivariate
    # ----------------------------------------------------
    plot = sales_by_store(df)
    save_plot("sales_by_store")

    plot = holiday_impact(df)
    save_plot("holiday_impact")

    # ----------------------------------------------------
    # Multivariate
    # ----------------------------------------------------
    plot = monthly_trend(df)
    save_plot("monthly_trend")

    print("\n(✓) -> EDA Pipeline Completed Successfully\n")
    print("\n==========( EDA Pipeline Ended )==========\n")


if __name__ == "__main__":
    run_eda_pipeline()