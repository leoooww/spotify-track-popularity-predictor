"""Model training, evaluation, and feature importance extraction.

Provides a registry of regression models, a unified training/evaluation
interface, and XGBoost feature importance helpers.
"""

import time
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model, tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


def get_model_registry() -> dict:
    """Return a dict of model_name → instantiated estimator."""
    return {
        "XGBoost": XGBRegressor(
            learning_rate=0.01,
            colsample_bytree=0.3,
            subsample=0.8,
            n_estimators=1000,
            reg_alpha=0.3,
            max_depth=10,
            gamma=1,
            verbosity=0,
        ),
        "Ridge": linear_model.Ridge(alpha=0.5),
        "Lasso": linear_model.Lasso(alpha=0.1),
        "DecisionTree": tree.DecisionTreeRegressor(),
        "BayesianRidge": linear_model.BayesianRidge(),
        "LinearRegression": linear_model.LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200),
    }


def split_data(
    df: pd.DataFrame,
    target: str = "popularity",
    test_size: float = 0.2,
    random_state: int = 0,
):
    """Split dataframe into train/test arrays."""
    X = df.drop(columns=[target]).values
    y = df[target].values
    feature_names = [c for c in df.columns if c != target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, feature_names


def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    """Fit a model and return MSE, R², and wall-clock time."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed = round(time.perf_counter() - t0, 3)
    return {
        "mse": round(float(mean_squared_error(y_test, y_pred)), 5),
        "r2": round(float(r2_score(y_test, y_pred)), 5),
        "time_s": elapsed,
    }


def evaluate_polynomial(
    degree: int, X_train, y_train, X_test, y_test
) -> dict:
    """Fit a polynomial regression of given degree and return metrics."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    t0 = time.perf_counter()
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    model = linear_model.LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    elapsed = round(time.perf_counter() - t0, 3)
    return {
        "mse": round(float(mean_squared_error(y_test, y_pred)), 5),
        "r2": round(float(r2_score(y_test, y_pred)), 5),
        "time_s": elapsed,
    }


def run_all_models(
    df: pd.DataFrame, target: str = "popularity"
) -> pd.DataFrame:
    """Train all registered models + polynomial regressions, return results."""
    X_train, X_test, y_train, y_test, feature_names = split_data(df, target)

    rows = []
    registry = get_model_registry()
    for name, model in registry.items():
        logger.info("Training %s...", name)
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        rows.append({"model": name, **metrics})

    for degree in [2, 3]:
        logger.info("Training Polynomial degree %d...", degree)
        metrics = evaluate_polynomial(degree, X_train, y_train, X_test, y_test)
        rows.append({"model": f"Polynomial_deg{degree}", **metrics})

    results = pd.DataFrame(rows)
    results = results[results["r2"] >= 0].sort_values("mse")
    return results


def xgboost_feature_importance(
    df: pd.DataFrame, target: str = "popularity"
) -> pd.DataFrame:
    """Train XGBoost and return a feature importance dataframe."""
    X_train, X_test, y_train, y_test, feature_names = split_data(df, target)
    model = get_model_registry()["XGBoost"]
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
