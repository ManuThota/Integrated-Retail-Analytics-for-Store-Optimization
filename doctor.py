"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          Integrated Retail Analytics — Environment Doctor                    ║
║                                                                              ║
║  Run this script after cloning the repo to verify your environment is        ║
║  set up correctly before executing the full pipeline.                        ║
║                                                                              ║
║  Usage:  python doctor.py                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import importlib.util

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# ANSI colour codes (fall back gracefully on Windows without ANSI support)
_ANSI = sys.stdout.isatty() or os.name != "nt"

GREEN  = "\033[92m" if _ANSI else ""
RED    = "\033[91m" if _ANSI else ""
YELLOW = "\033[93m" if _ANSI else ""
CYAN   = "\033[96m" if _ANSI else ""
BOLD   = "\033[1m"  if _ANSI else ""
RESET  = "\033[0m"  if _ANSI else ""

PASS = f"{GREEN}  [✔] FOUND    {RESET}"
FAIL = f"{RED}  [✘] MISSING  {RESET}"
INFO = f"{CYAN}  [i]          {RESET}"

# Base directory (wherever this file lives)
BASE = os.path.dirname(os.path.abspath(__file__))


def section(title: str) -> None:
    width = 70
    print()
    print(f"{BOLD}{CYAN}{'─' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * width}{RESET}")


def check_path(rel_path: str, label: str | None = None) -> bool:
    """Return True if *rel_path* (relative to BASE) exists."""
    full = os.path.join(BASE, rel_path)
    display = label or rel_path
    if os.path.exists(full):
        print(f"{PASS}{display}")
        return True
    else:
        print(f"{FAIL}{display}")
        return False


def fix_hint(message: str) -> None:
    print(f"{YELLOW}       ↳ {message}{RESET}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Virtual Environment
# ──────────────────────────────────────────────────────────────────────────────

def check_venv() -> bool:
    section("1 · Virtual Environment")

    venv_ok = check_path("venv", "venv/  (virtual environment folder)")
    if not venv_ok:
        fix_hint("Create it:  python -m venv venv")
        fix_hint("Then activate — see README.md § 'Create Virtual Environment'")

    # Warn if the script is NOT running inside a venv
    in_venv = (
        hasattr(sys, "real_prefix")                      # virtualenv
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)  # venv
    )
    if in_venv:
        print(f"{PASS}Running inside an active virtual environment")
    else:
        print(f"{FAIL}NOT running inside the virtual environment")
        fix_hint("Activate it first:")
        fix_hint("  Windows : venv\\Scripts\\activate")
        fix_hint("  Mac/Linux: source venv/bin/activate")
        fix_hint("See README.md § 'Activate Virtual Environment'")

    return venv_ok and in_venv


# ──────────────────────────────────────────────────────────────────────────────
# 2. Python Dependencies
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_PACKAGES = {
    "pandas":        "pandas",
    "numpy":         "numpy",
    "matplotlib":    "matplotlib",
    "seaborn":       "seaborn",
    "sklearn":       "scikit-learn",
    "joblib":        "joblib",
}


def check_dependencies() -> bool:
    section("2 · Python Dependencies  (requirements.txt)")

    all_ok = True
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        found = importlib.util.find_spec(import_name) is not None
        label = f"{pip_name:<20}  (import {import_name})"
        if found:
            print(f"{PASS}{label}")
        else:
            print(f"{FAIL}{label}")
            fix_hint(f"Install:  pip install {pip_name}")
            all_ok = False

    if not all_ok:
        print()
        fix_hint("Or install everything at once:  pip install -r requirements.txt")
        fix_hint("See README.md § 'Install Dependencies'")

    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# 3. Datasets
# ──────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "data/raw/sales.csv":           "Raw sales data",
    "data/raw/stores.csv":          "Raw stores data",
    "data/raw/features.csv":        "Raw features / external factors data",
}


def check_datasets() -> bool:
    section("3 · Datasets  (data/raw/)")

    all_ok = True
    for rel, desc in DATASETS.items():
        ok = check_path(rel, f"{os.path.basename(rel):<25}  — {desc}")
        if not ok:
            all_ok = False

    if not all_ok:
        print()
        fix_hint("Download datasets from:")
        fix_hint("  https://drive.google.com/drive/folders/1z9mPgev0LU23aKAAzgVVpqJ82NQDRapN")
        fix_hint("Place ALL three CSV files inside  data/raw/")
        fix_hint("See README.md § 'Place datasets'")

    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# 4. Source Modules
# ──────────────────────────────────────────────────────────────────────────────

SRC_FILES = {
    # Config
    "src/config/config.py":                           "Config — paths & hyperparameters",
    # Data ingestion
    "src/data_ingestion/load_data.py":                "Data Ingestion — load raw CSVs",
    # Data preprocessing
    "src/data_preprocessing/cleaning.py":             "Preprocessing — MarkDown imputation",
    "src/data_preprocessing/merging.py":              "Preprocessing — 3-way dataset merge",
    "src/data_preprocessing/transformation.py":       "Preprocessing — date parsing, log1p",
    "src/data_preprocessing/saving.py":               "Preprocessing — save interim/processed",
    # EDA
    "src/eda/univariate.py":                          "EDA — distributions & text summary",
    "src/eda/bivariate.py":                           "EDA — sales by type, holiday effect",
    "src/eda/multivariate.py":                        "EDA — correlation heatmap",
    "src/eda/visualization.py":                       "EDA — shared plot helpers",
    # Anomaly detection
    "src/anomaly_detection/statistical.py":           "Anomaly Detection — IQR & Z-score",
    "src/anomaly_detection/time_series.py":           "Anomaly Detection — rolling baseline",
    # Feature engineering
    "src/feature_engineering/feature_builder.py":     "Feature Engineering — time features",
    "src/feature_engineering/encoding.py":            "Feature Engineering — One-Hot Encoding",
    "src/feature_engineering/feature_selection.py":   "Feature Engineering — selection & split",
    # Preprocessing
    "src/preprocessing/scaler.py":                    "Preprocessing — StandardScaler",
    # Model building
    "src/model_building/train.py":                    "Model Building — RandomForest training",
    "src/model_building/evaluate.py":                 "Model Building — RMSE/MAE/R² metrics",
    "src/model_building/tune.py":                     "Model Building — hyperparameter tuning",
    # Segmentation
    "src/segmentation/clustering.py":                 "Segmentation — K-Means clustering",
    "src/segmentation/evaluation.py":                 "Segmentation — silhouette & elbow",
    # Market basket
    "src/market_basket/association.py":               "Market Basket — dept correlation",
    # External factors
    "src/external_factors/analysis.py":               "External Factors — CPI/fuel/unemployment",
    # Personalization
    "src/personalization/strategy.py":                "Personalization — cluster-based strategy",
    # Explainability
    "src/explainability/feature_importance.py":       "Explainability — MDI + permutation",
    # Utils
    "src/utils/logger.py":                            "Utils — centralised logging",
    "src/utils/helpers.py":                           "Utils — helper functions",
}


def check_src_modules() -> bool:
    section("4 · Source Modules  (src/)")

    all_ok = True
    for rel, desc in SRC_FILES.items():
        ok = check_path(rel, f"{rel:<50}  {desc}")
        if not ok:
            all_ok = False

    if not all_ok:
        print()
        fix_hint("Some source files are missing — the repo may be incomplete.")
        fix_hint("Try:  git status  (to see untracked/deleted files)")
        fix_hint("Or re-clone:  git clone https://github.com/ManuThota/Integrated-Retail-Analytics-for-Store-Optimization.git")

    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# 5. Pipeline Files
# ──────────────────────────────────────────────────────────────────────────────

PIPELINES = {
    "src/pipeline/data_pipeline.py":  "Data Pipeline  — stages 1–4 (ingest → save)",
    "src/pipeline/train_pipeline.py": "Train Pipeline — stages 5–8 (engineer → eval)",
    "src/pipeline/full_pipeline.py":  "Full Pipeline  — orchestrates both pipelines",
    "main.py":                        "main.py        — entry-point (run this!)",
}


def check_pipelines() -> bool:
    section("5 · Pipelines")

    all_ok = True
    for rel, desc in PIPELINES.items():
        ok = check_path(rel, f"{os.path.basename(rel):<30}  {desc}")
        if not ok:
            all_ok = False

    if not all_ok:
        print()
        fix_hint("Pipeline files are missing — the project cannot run.")
        fix_hint("See README.md § 'Pipeline Stages' for expected files.")

    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

def summary(results: dict[str, bool]) -> None:
    section("Summary")

    total   = len(results)
    passed  = sum(results.values())
    failed  = total - passed

    for check_name, ok in results.items():
        icon = f"{GREEN}✔{RESET}" if ok else f"{RED}✘{RESET}"
        print(f"  {icon}  {check_name}")

    print()
    if failed == 0:
        print(
            f"{BOLD}{GREEN}  ✔ All {total} checks passed — you're good to go!{RESET}\n"
            f"{INFO}Run the pipeline with:  python main.py"
        )
    else:
        print(
            f"{BOLD}{RED}  ✘ {failed}/{total} check(s) failed.{RESET}\n"
            f"{YELLOW}  Please fix the issues listed above, then re-run:  python doctor.py{RESET}\n"
            f"{YELLOW}  Refer to README.md for detailed setup instructions.{RESET}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(
        f"\n{BOLD}{CYAN}"
        "╔══════════════════════════════════════════════════════════════════════╗\n"
        "║     Integrated Retail Analytics — Environment Doctor                 ║\n"
        "╚══════════════════════════════════════════════════════════════════════╝"
        f"{RESET}"
    )
    print(f"{INFO}Checking your environment… (base dir: {BASE})")

    results = {
        "Virtual Environment": check_venv(),
        "Python Dependencies": check_dependencies(),
        "Datasets           ": check_datasets(),
        "Source Modules     ": check_src_modules(),
        "Pipeline Files     ": check_pipelines(),
    }

    summary(results)
    print()

    # Exit with a non-zero code so CI/CD pipelines can detect failure
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
