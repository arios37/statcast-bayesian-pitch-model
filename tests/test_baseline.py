"""
Tests for ML baseline models.

Verify that baselines run on synthetic data and return
expected output format.
"""

import numpy as np
import pandas as pd
import pytest

from src.baseline import (
    FEATURE_COLS,
    BaselineResult,
    fit_gradient_boosting,
    fit_linear_regression,
    run_all_baselines,
)


@pytest.fixture
def baseline_df():
    """Synthetic feature-engineered dataframe for baseline testing."""
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        "release_speed": np.random.normal(0, 1, n),
        "pfx_x": np.random.normal(0, 1, n),
        "pfx_z": np.random.normal(0, 1, n),
        "plate_x": np.random.normal(0, 1, n),
        "plate_z": np.random.normal(0, 1, n),
        "count_leverage": np.random.normal(0, 1, n),
        "platoon_adv": np.random.choice([0, 1], n),
        "stuff_composite": np.random.normal(0, 1, n),
        "total_movement": np.random.normal(0, 1, n),
        "delta_run_exp": np.random.normal(0, 0.15, n),
    })


class TestLinearRegression:
    def test_returns_result(self, baseline_df):
        result = fit_linear_regression(baseline_df, cv_folds=3)
        assert isinstance(result, BaselineResult)
        assert result.name == "Linear Regression"

    def test_metrics_reasonable(self, baseline_df):
        result = fit_linear_regression(baseline_df, cv_folds=3)
        assert result.rmse >= 0
        assert result.mae >= 0
        assert result.cv_rmse_mean >= 0

    def test_feature_importance_keys(self, baseline_df):
        result = fit_linear_regression(baseline_df, cv_folds=3)
        assert result.feature_importance is not None
        for col in FEATURE_COLS:
            assert col in result.feature_importance


class TestGradientBoosting:
    def test_returns_result(self, baseline_df):
        result = fit_gradient_boosting(baseline_df, cv_folds=3, n_estimators=10)
        assert isinstance(result, BaselineResult)

    def test_importance_sums_to_one(self, baseline_df):
        result = fit_gradient_boosting(baseline_df, cv_folds=3, n_estimators=10)
        total = sum(result.feature_importance.values())
        assert abs(total - 1.0) < 0.01


class TestRunAllBaselines:
    def test_returns_dataframe(self, baseline_df):
        comparison = run_all_baselines(baseline_df, cv_folds=3)
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3

    def test_expected_columns(self, baseline_df):
        comparison = run_all_baselines(baseline_df, cv_folds=3)
        expected = ["Model", "RMSE", "MAE", "R-squared", "CV RMSE (mean)"]
        for col in expected:
            assert col in comparison.columns
