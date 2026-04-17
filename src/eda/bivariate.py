"""
bivariate.py

Analyzes relationships between variables:
- Store vs Sales
- Holiday vs Sales
"""

import matplotlib.pyplot as plt
import seaborn as sns


def sales_by_store(df):
    """
    Average sales per store.
    """

    store_sales = df.groupby("Store")["Weekly_Sales"].mean()

    plt.figure(figsize=(12, 6))
    store_sales.plot()

    plt.title("Average Sales by Store")
    plt.xlabel("Store")
    plt.ylabel("Average Sales")

    print("(✓) -> Store-wise sales plot created")

    return plt


def holiday_impact(df):
    """
    Sales comparison between holiday and non-holiday.
    """

    plt.figure(figsize=(8, 5))

    sns.boxplot(x="IsHoliday", y="Weekly_Sales", data=df)

    plt.title("Holiday Impact on Sales")

    print("(✓) -> Holiday impact plot created")

    return plt