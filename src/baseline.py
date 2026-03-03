"""
ML baseline models for pitch-level expected run value.

Provides Linear Regression, Random Forest, and XGBoost baselines
to benchmark against the Bayesian hierarchical model.

The point isn't that these are worse. The point is that they give
you point estimates. The Bayesian model gives you distributions.
Same prediction, fundamentally different information content.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

# Features used by all baseline models
FEATURE_COLS = [
    "release_speed",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "count_leverage",
    "platoon_adv",
    "stuff_composite",
    "total_movement",
]

TARGET_COL = "delta_run_exp"


@dataclass
class BaselineResult:
    """Container for baseline model evaluation results."""

    name: str
    rmse: float
    mae: float
    r2: float
    cv_rmse_mean: float
    cv_rmse_std: float
    feature_importance: dict[str, float] | None = None


def _get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract X, y arrays from the dataframe."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    available = [c for c in feature_cols if c in df.columns]
    if len(available) < len(feature_cols):
        missing = set(feature_cols) - set(available)
        logger.warning("Missing features: %s", missing)

    features = df[available].values
    target = df[TARGET_COL].values
    return features, target, available


def fit_linear_regression(
    df: pd.DataFrame,
    cv_folds: int = 5,
) -> BaselineResult:
    """
    OLS linear regression baseline.

    The simplest possible model. No regularization, no hierarchy,
    no uncertainty. Just a hyperplane through feature space.
    """
    features, target, cols = _get_feature_matrix(df)

    model = LinearRegression()
    model.fit(features, target)
    y_pred = model.predict(features)

    # Cross-validated RMSE
    cv_scores = cross_val_score(
        model, features, target, cv=cv_folds,
        scoring="neg_root_mean_squared_error",
    )

    importance = dict(zip(cols, model.coef_, strict=True))

    result = BaselineResult(
        name="Linear Regression",
        rmse=float(np.sqrt(mean_squared_error(target, y_pred))),
        mae=float(mean_absolute_error(target, y_pred)),
        r2=float(r2_score(target, y_pred)),
        cv_rmse_mean=float(-cv_scores.mean()),
        cv_rmse_std=float(cv_scores.std()),
        feature_importance=importance,
    )
    logger.info("%s: RMSE=%.4f, R2=%.4f, CV-RMSE=%.4f", result.name, result.rmse, result.r2, result.cv_rmse_mean)
    return result


def fit_random_forest(
    df: pd.DataFrame,
    cv_folds: int = 5,
    n_estimators: int = 200,
    max_depth: int = 12,
    random_state: int = 42,
) -> BaselineResult:
    """
    Random forest regressor.

    Non-linear, handles interactions automatically, feature importance
    comes free. But still point estimates, no uncertainty.
    """
    features, target, cols = _get_feature_matrix(df)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(features, target)
    y_pred = model.predict(features)

    cv_scores = cross_val_score(
        model, features, target, cv=cv_folds,
        scoring="neg_root_mean_squared_error",
    )

    importance = dict(zip(cols, model.feature_importances_, strict=True))

    result = BaselineResult(
        name="Random Forest",
        rmse=float(np.sqrt(mean_squared_error(target, y_pred))),
        mae=float(mean_absolute_error(target, y_pred)),
        r2=float(r2_score(target, y_pred)),
        cv_rmse_mean=float(-cv_scores.mean()),
        cv_rmse_std=float(cv_scores.std()),
        feature_importance=importance,
    )
    logger.info("%s: RMSE=%.4f, R2=%.4f, CV-RMSE=%.4f", result.name, result.rmse, result.r2, result.cv_rmse_mean)
    return result


def fit_gradient_boosting(
    df: pd.DataFrame,
    cv_folds: int = 5,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> BaselineResult:
    """
    Gradient boosting regressor (sklearn, not XGBoost -- fewer deps).

    Usually the strongest point-estimate baseline. If this beats the
    Bayesian model on raw RMSE, that's fine. The Bayesian model still
    gives you something this can't: calibrated uncertainty per pitcher.
    """
    features, target, cols = _get_feature_matrix(df)

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(features, target)
    y_pred = model.predict(features)

    cv_scores = cross_val_score(
        model, features, target, cv=cv_folds,
        scoring="neg_root_mean_squared_error",
    )

    importance = dict(zip(cols, model.feature_importances_, strict=True))

    result = BaselineResult(
        name="Gradient Boosting",
        rmse=float(np.sqrt(mean_squared_error(target, y_pred))),
        mae=float(mean_absolute_error(target, y_pred)),
        r2=float(r2_score(target, y_pred)),
        cv_rmse_mean=float(-cv_scores.mean()),
        cv_rmse_std=float(cv_scores.std()),
        feature_importance=importance,
    )
    logger.info("%s: RMSE=%.4f, R2=%.4f, CV-RMSE=%.4f", result.name, result.rmse, result.r2, result.cv_rmse_mean)
    return result


def run_all_baselines(
    df: pd.DataFrame,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """
    Run all baseline models and return a comparison table.

    This is what goes in the notebook: one function call, clean output.
    """
    results = [
        fit_linear_regression(df, cv_folds=cv_folds),
        fit_random_forest(df, cv_folds=cv_folds),
        fit_gradient_boosting(df, cv_folds=cv_folds),
    ]

    comparison = pd.DataFrame([
        {
            "Model": r.name,
            "RMSE": r.rmse,
            "MAE": r.mae,
            "R-squared": r.r2,
            "CV RMSE (mean)": r.cv_rmse_mean,
            "CV RMSE (std)": r.cv_rmse_std,
        }
        for r in results
    ])

    return comparison
