"""
visualization.py

Utility functions for:
- Saving plots
- Standardizing visualization outputs
"""

import os
import matplotlib.pyplot as plt
from src.config.config import FIGURES_DIR


def save_plot(filename: str) -> None:
    """
    Saves the current matplotlib plot to the figures directory.

    Args:
        filename (str): Name of the file (without extension)
    """

    # Ensure directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    file_path = FIGURES_DIR / f"{filename}.png"

    plt.savefig(file_path, bbox_inches="tight")
    plt.close()

    print(f"(✓) -> Plot saved: {file_path}")