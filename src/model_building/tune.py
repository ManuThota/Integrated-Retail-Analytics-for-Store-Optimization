"""
tune.py

Handles:
- Hyperparameter tuning using RandomizedSearchCV
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from src.config.config import RANDOM_STATE


def tune_random_forest(X_train, y_train):
    """
    Performs hyperparameter tuning.

    Returns:
        best model
    """

    param_dist = {
        "n_estimators": [100, 150],
        "max_depth": [None, 20, 30],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    base_model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=5,
        cv=3,
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    print("(✓) -> Hyperparameter tuning completed")
    print(f"Best Parameters: {random_search.best_params_}")

    best_model = random_search.best_estimator_

    return best_model