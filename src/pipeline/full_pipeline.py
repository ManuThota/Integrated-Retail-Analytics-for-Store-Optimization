"""
Full Pipeline Module
======================
Entry-point for the complete Integrated Retail Analytics pipeline.

Delegates to the two sub-pipelines and prints a live progress
summary to the terminal so you can track every stage as it runs.
"""

import logging
import time

from src.utils.helpers import format_duration
from src.pipeline.data_pipeline import run_data_pipeline
from src.pipeline.train_pipeline import run_train_pipeline

logger = logging.getLogger(__name__)

_W     = 58          # banner width
_SEP   = "═" * _W   # thick separator


def _header() -> None:
    """Print the pipeline start banner."""
    print()
    print(_SEP)
    print("  INTEGRATED RETAIL ANALYTICS — PIPELINE START")
    print(_SEP)
    print("  Model  : Random Forest Regressor")
    print("  Target : log1p(Weekly_Sales)")
    print("  Encode : One-Hot (Type A/B/C → Type_B, Type_C)")
    print(_SEP)


def _footer(metrics: dict, cv_result: dict | None, elapsed: float) -> None:
    """Print the final results banner."""
    print()
    print(_SEP)
    print("  PIPELINE COMPLETE")
    print(_SEP)
    if cv_result:
        print(f"  CV R²  (mean)  :  {cv_result['mean']:.4f}")
        print(f"  CV R²  (± std) :  {cv_result['std']:.4f}")
    print(f"  Test RMSE      :  {metrics['rmse']:.4f}")
    print(f"  Test MAE       :  {metrics['mae']:.4f}")
    print(f"  Test R²        :  {metrics['r2']:.4f}")
    print(f"  Total time     :  {format_duration(elapsed)}")
    print(_SEP)
    print()
    print("  Outputs written:")
    print("    models/random_forest.pkl")
    print("    models/scaler.pkl")
    print("    reports/metrics.json")
    print("    reports/insights.txt")
    print("    reports/cv_results.txt")
    print("    reports/figures/  (6 plots)")
    print("    logs/pipeline.log")
    print(_SEP)
    print()


def run_pipeline(run_cv: bool = True) -> None:
    """Execute the full end-to-end Integrated Retail Analytics pipeline.

    Args:
        run_cv: Whether to run 5-fold cross-validation before final training.
                Default ``True``.
    """
    _header()
    t_start = time.time()

    # ── Data stages (1–4) ──────────────────────────────────────────────────────
    df_processed = run_data_pipeline()

    # ── Training stages (5–8) ─────────────────────────────────────────────────
    result = run_train_pipeline(df=df_processed, run_cv=run_cv)

    # ── Summary ───────────────────────────────────────────────────────────────
    _footer(
        metrics=result["metrics"],
        cv_result=result.get("cv"),
        elapsed=time.time() - t_start,
    )
