"""
Central Configuration Module
==============================
Single source of truth for every path, hyperparameter, and column definition
used across the project.  Import from here — never hard-code paths in modules.

Directory layout this file assumes
────────────────────────────────────
Integrated Retail Analytics for Store Optimization/
├── data/raw/            ← original CSVs
├── data/interim/        ← merged, un-transformed CSVs
├── data/processed/      ← fully engineered CSVs
├── models/              ← serialised model & scaler artefacts
├── reports/
│   └── figures/         ← all generated plots
├── logs/
└── src/config/config.py ← THIS FILE  (2 directories up = project root)
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# ROOT & DIRECTORY PATHS
# ══════════════════════════════════════════════════════════════════════════════

#: Absolute path to the project root (parent of the ``src/`` directory).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
INTERIM_DIR   = DATA_DIR / "interim"        # merged but un-transformed
PROCESSED_DIR = DATA_DIR / "processed"      # cleaned, engineered, model-ready

# ── Raw file paths ─────────────────────────────────────────────────────────────
SALES_CSV    = RAW_DIR / "sales.csv"         # weekly store-dept sales
STORES_CSV   = RAW_DIR / "stores.csv"        # store type & size metadata
FEATURES_CSV = RAW_DIR / "features.csv"      # macro-economic + promotion data

# ── Checkpoint file paths ──────────────────────────────────────────────────────
INTERIM_MERGED_CSV = INTERIM_DIR / "merged_data.csv"   # post-merge checkpoint
PROCESSED_CSV      = PROCESSED_DIR / "final_data.csv"  # post-transform checkpoint

# ── Model artefacts ────────────────────────────────────────────────────────────
MODELS_DIR  = PROJECT_ROOT / "models"
MODEL_PATH  = MODELS_DIR / "random_forest.pkl"   # final trained Random Forest
SCALER_PATH = MODELS_DIR / "scaler.pkl"          # fitted StandardScaler

# ── Reports ────────────────────────────────────────────────────────────────────
REPORTS_DIR  = PROJECT_ROOT / "reports"
FIGURES_DIR  = REPORTS_DIR / "figures"       # all .png plots saved here
METRICS_JSON = REPORTS_DIR / "metrics.json"  # machine-readable metrics
INSIGHTS_TXT = REPORTS_DIR / "insights.txt"  # human-readable text summary

# ── Logging ────────────────────────────────────────────────────────────────────
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "pipeline.log"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

N_ESTIMATORS  = 100   # number of trees (notebook default)
RANDOM_STATE  = 42    # global reproducibility seed
TEST_SIZE     = 0.2   # proportion of data held out for final evaluation
CV_FOLDS      = 5     # k-fold cross-validation folds (notebook: 5)
N_ITER_SEARCH = 20    # iterations for RandomizedSearchCV

#: Parameter distributions / grid for hyperparameter search.
#: Used by ``src/model_building/tune.py``.
PARAM_DISTRIBUTIONS = {
    "n_estimators":      [50, 100, 150, 200],
    "max_depth":         [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2"],
    "bootstrap":         [True, False],
}

#: Focused grid for a follow-up GridSearchCV after RandomizedSearch
#: has identified a promising region of the parameter space.
FINE_TUNE_GRID = {
    "n_estimators":      [100, 150],
    "max_depth":         [None, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
}


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

#: Target column — log1p-transformed Weekly_Sales.
TARGET_COL = "Weekly_Sales"

#: Base numeric/boolean feature columns used by the model.
#: NOTE: 'Type' is intentionally excluded here because it is converted to
#:       dummy columns (Type_B, Type_C, …) via One-Hot Encoding in
#:       ``src/feature_engineering/encoding.py``.  Those OHE columns are
#:       then appended dynamically in feature_selection.py.
FEATURE_COLS = [
    "Store",
    "Dept",
    "IsHoliday",       # bool → int (0/1) during encoding step
    "Temperature",
    "Fuel_Price",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "CPI",
    "Unemployment",
    "Size",
    "Year",
    "Month",
    "Week",
    # Type_B, Type_C (OHE dummy columns) are added at runtime
]

#: Columns to fill NaN with 0 (missing markdown = no promotion active).
MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

#: Categorical columns to apply One-Hot Encoding on before model training.
OHE_COLUMNS = ["Type"]

#: Columns to aggregate for K-Means store segmentation.
CLUSTER_FEATURE_COLS = [
    "Weekly_Sales",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "CPI", "Unemployment", "Size",
]
