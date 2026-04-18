"""
Main Entry-Point
=================
Run the full Integrated Retail Analytics pipeline:

    python main.py

To skip cross-validation (faster re-runs):

    python -c "from src.pipeline.full_pipeline import run_pipeline; run_pipeline(run_cv=False)"
"""

from src.utils.logger import get_logger
from src.pipeline.full_pipeline import run_pipeline

logger = get_logger(__name__)

if __name__ == "__main__":
    run_pipeline(run_cv=True)
