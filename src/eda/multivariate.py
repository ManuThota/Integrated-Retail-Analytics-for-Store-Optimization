"""
multivariate.py

Handles multi-variable analysis:
- Time trends
- Seasonal patterns
"""

import matplotlib.pyplot as plt


def monthly_trend(df):
    """
    Monthly sales trend.
    """

    df["Month"] = df["Date"].dt.month

    monthly_sales = df.groupby("Month")["Weekly_Sales"].mean()

    plt.figure(figsize=(10, 5))
    monthly_sales.plot(marker="o")

    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Average Sales")

    print("(✓) -> Monthly trend plot created")

    return plt