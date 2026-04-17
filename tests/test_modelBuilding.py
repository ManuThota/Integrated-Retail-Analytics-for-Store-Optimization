"""
test_modelBuilding.py

Tests the full model building pipeline:
- Data loading (processed data)
- Train-test split
- Model training (Random Forest)
- Evaluation
- Cross-validation
- Hyperparameter tuning

Ensures:
- Model trains successfully
- Predictions work
- Metrics are valid
"""

# ============================================================
# Imports
# ============================================================

import sys
import os
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from src.config.config import PROCESSED_DATA_DIR

from src.feature_engineering.encoding import encode_categorical
from src.feature_engineering.feature_builder import (
    create_time_features,
    create_ml_dataset
)

from src.model_building.train import split_data, train_random_forest
from src.model_building.evaluate import evaluate_model, cross_validate_model
from src.model_building.tune import tune_random_forest


# ============================================================
# Test Function
# ============================================================

def test_model_pipeline():
    """
    Runs full ML pipeline and validates results.
    """

    try:
        print("\n==========( Model Building Test Started )==========\n")

        # ----------------------------------------------------
        # Step 1: Load Data
        # ----------------------------------------------------
        file_path = PROCESSED_DATA_DIR / "processed_data.csv"

        if not file_path.exists():
            raise FileNotFoundError("(✕) -> Run preprocessing first")

        df = pd.read_csv(file_path, parse_dates=["Date"])

        print(f"(✓) -> Data loaded | Shape: {df.shape}")

        # ----------------------------------------------------
        # Step 2: Feature Engineering
        # ----------------------------------------------------
        df = encode_categorical(df)
        df = create_time_features(df)

        df_ml = create_ml_dataset(df)

        assert "Weekly_Sales" in df_ml.columns, "(✕) -> Target column missing"

        print("(✓) -> Feature engineering completed")

        # ----------------------------------------------------
        # Step 3: Train-Test Split
        # ----------------------------------------------------
        X_train, X_test, y_train, y_test = split_data(df_ml)

        assert X_train.shape[0] > 0, "(✕) -> Training data is empty"

        # ----------------------------------------------------
        # Step 4: Train Model
        # ----------------------------------------------------
        model = train_random_forest(X_train, y_train)

        assert model is not None, "(✕) -> Model training failed"

        # ----------------------------------------------------
        # Step 5: Evaluate Model
        # ----------------------------------------------------
        metrics = evaluate_model(model, X_test, y_test)

        assert metrics["R2"] > 0, "(✕) -> Model performance too low"

        # ----------------------------------------------------
        # Step 6: Cross Validation
        # ----------------------------------------------------
        cv_score = cross_validate_model(model, X_train, y_train)

        assert cv_score > 0, "(✕) -> Cross-validation score invalid"

        # ----------------------------------------------------
        # Step 7: Hyperparameter Tuning
        # ----------------------------------------------------
        best_model = tune_random_forest(X_train, y_train)

        assert best_model is not None, "(✕) -> Tuning failed"

        # ----------------------------------------------------
        # Step 8: Evaluate Tuned Model
        # ----------------------------------------------------
        tuned_metrics = evaluate_model(best_model, X_test, y_test)

        print("\n(✓) -> Tuned Model Evaluation Completed")

        # ----------------------------------------------------
        # Final Success
        # ----------------------------------------------------
        print("\n(✓) -> ALL MODEL BUILDING TESTS PASSED SUCCESSFULLY!\n")

    except Exception as e:
        print("\n(✕) -> TEST FAILED!\n")
        print("Error:", str(e))
        print("\nTraceback:")
        traceback.print_exc()


# ============================================================
# Run Test
# ============================================================

if __name__ == "__main__":
    test_model_pipeline()