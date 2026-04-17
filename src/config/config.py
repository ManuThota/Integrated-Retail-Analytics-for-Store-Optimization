"""
config.py

This file contains all the configuration settings for the project.
It centralizes paths, constants, and model parameters to ensure:
- Reproducibility
- Maintainability
- No hardcoding across the project
"""

import os
from pathlib import Path

# ============================================================
# Project Root Path
# ============================================================

# Get the absolute path of the project root directory
# (__file__ → config.py → src/config → go 2 levels up)
BASE_DIR = Path(__file__).resolve().parents[2]

# ============================================================
# Data Paths
# ============================================================

DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# File names
SALES_DATA_PATH = RAW_DATA_DIR / "sales.csv"
STORES_DATA_PATH = RAW_DATA_DIR / "stores.csv"
FEATURES_DATA_PATH = RAW_DATA_DIR / "features.csv"

# ============================================================
# Model Paths
# ============================================================

MODEL_DIR = BASE_DIR / "models"

RANDOM_FOREST_MODEL_PATH = MODEL_DIR / "random_forest.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# ============================================================
# Reports Paths
# ============================================================

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

METRICS_PATH = REPORTS_DIR / "metrics.json"
INSIGHTS_PATH = REPORTS_DIR / "insights.txt"

# ============================================================
# General Settings
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ============================================================
# Feature Engineering Settings
# ============================================================

DATE_COLUMN = "Date"
TARGET_COLUMN = "Weekly_Sales"

# Time-based features
TIME_FEATURES = ["Year", "Month", "Week"]
