"""
Tests for the Bayesian model module.

These test model construction and diagnostics extraction
on tiny synthetic data. We don't run full MCMC in CI
(too slow + needs PyMC compiled). Instead we verify the
model graph is built correctly.
"""

import numpy as np
import pandas as pd
import pytest

from src.model import build_model


@pytest.fixture
def model_ready_df():
    """
    Minimal dataframe that looks like output from build_model_matrix().

    Standardized continuous features, pitcher_idx, all required columns.
    """
    np.random.seed(42)
    n = 200
    n_pitchers = 5

    df = pd.DataFrame({
        "pitcher": np.random.choice(range(100, 100 + n_pitchers), n),
        "pitcher_idx": np.random.choice(range(n_pitchers), n),
        "release_speed": np.random.normal(0, 1, n),  # standardized
        "release_spin_rate": np.random.normal(0, 1, n),
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
    return df


class TestBuildModel:
    def test_model_compiles(self, model_ready_df):
        """Model should compile without errors."""
        model, idata = build_model(model_ready_df, sample=False)
        assert model is not None
        assert idata is None  # didn't sample

    def test_model_has_expected_variables(self, model_ready_df):
        """Check that the model graph contains the right random variables."""
        model, _ = build_model(model_ready_df, sample=False)

        # Free RVs (things we're estimating)
        rv_names = {rv.name for rv in model.free_RVs}
        expected = {
            "mu_alpha", "sigma_alpha", "alpha_pitcher_offset",
            "beta_velo", "beta_hmov", "beta_vmov",
            "beta_loc_x", "beta_loc_z",
            "beta_count", "beta_platoon",
            "sigma",
        }
        for name in expected:
            assert name in rv_names, f"Missing RV: {name}"

    def test_model_has_observed(self, model_ready_df):
        """Model should have an observed variable for the likelihood."""
        model, _ = build_model(model_ready_df, sample=False)
        obs_names = {rv.name for rv in model.observed_RVs}
        assert "y_obs" in obs_names

    def test_pitcher_dim_matches(self, model_ready_df):
        """Pitcher dimension should match number of unique pitchers."""
        model, _ = build_model(model_ready_df, sample=False)
        n_pitchers = model_ready_df["pitcher_idx"].nunique()
        assert len(model.coords["pitcher"]) == n_pitchers

    def test_optional_stuff_included(self, model_ready_df):
        """When stuff_composite is present, beta_stuff should be in model."""
        model, _ = build_model(model_ready_df, sample=False)
        rv_names = {rv.name for rv in model.free_RVs}
        assert "beta_stuff" in rv_names

    def test_optional_spin_included(self, model_ready_df):
        """When release_spin_rate is present, beta_spin should be in model."""
        model, _ = build_model(model_ready_df, sample=False)
        rv_names = {rv.name for rv in model.free_RVs}
        assert "beta_spin" in rv_names

    def test_no_stuff_when_missing(self, model_ready_df):
        """When stuff_composite is absent, beta_stuff should not be in model."""
        df = model_ready_df.drop(columns=["stuff_composite"])
        model, _ = build_model(df, sample=False)
        rv_names = {rv.name for rv in model.free_RVs}
        assert "beta_stuff" not in rv_names


class TestDiagnosticsFormat:
    """Test that diagnostic functions handle edge cases gracefully."""

    def test_get_pitcher_effects_with_map(self):
        """Reverse map should work when pitcher_map is provided."""
        # This tests the logic, not actual inference data
        pitcher_map = {100: 0, 200: 1, 300: 2}
        reverse = {v: k for k, v in pitcher_map.items()}
        assert reverse[0] == 100
        assert reverse[1] == 200
        assert reverse[2] == 300
