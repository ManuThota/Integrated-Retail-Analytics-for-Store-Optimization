"""
univariate.py

Performs univariate analysis:
- Distribution of sales
"""

import matplotlib.pyplot as plt
import seaborn as sns


def sales_distribution(df):
    """
    Plots distribution of Weekly Sales.
    """

    plt.figure(figsize=(10, 5))

    sns.histplot(df["Weekly_Sales"], bins=50, kde=True)

    plt.title("Sales Distribution")
    plt.xlabel("Weekly Sales")
    plt.ylabel("Frequency")

    print("(✓) -> Sales distribution plot created")

    return plt