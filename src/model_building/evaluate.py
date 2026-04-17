"""
evaluate.py

Handles:
- Model evaluation
- Cross-validation
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score


def evaluate_model(model, X_test, y_test):
    """
    Evaluates model performance.

    Returns:
        dict of metrics
    """

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("(✓) -> Model evaluation completed")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.4f}")

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


def cross_validate_model(model, X_train, y_train):
    """
    Performs cross-validation.

    Returns:
        mean CV score
    """

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    print("(✓) -> Cross-validation completed")
    print(f"Scores: {scores}")
    print(f"Average CV Score: {scores.mean():.4f}")

    return scores.mean()