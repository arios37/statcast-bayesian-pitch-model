"""
Tests for feature engineering pipeline.

These validate the logic, not the data. The model is only as
good as the features going in.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    COUNT_LEVERAGE,
    add_base_state,
    add_count_leverage,
    add_location_zone,
    add_platoon_advantage,
    add_stuff_composite,
    create_pitcher_index,
)


@pytest.fixture
def sample_df():
    """Minimal Statcast-like dataframe for testing."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "pitcher": np.random.choice([100, 200, 300, 400], n),
        "batter": np.random.choice([500, 600, 700], n),
        "p_throws": np.random.choice(["R", "L"], n),
        "stand": np.random.choice(["R", "L"], n),
        "release_speed": np.random.normal(93, 5, n),
        "release_spin_rate": np.random.normal(2300, 300, n),
        "pfx_x": np.random.normal(0, 8, n),
        "pfx_z": np.random.normal(0, 8, n),
        "plate_x": np.random.normal(0, 0.8, n),
        "plate_z": np.random.normal(2.5, 0.7, n),
        "balls": np.random.choice([0, 1, 2, 3], n),
        "strikes": np.random.choice([0, 1, 2], n),
        "on_1b": np.random.choice([0, 1], n),
        "on_2b": np.random.choice([0, 1], n),
        "on_3b": np.random.choice([0, 1], n),
        "pitch_type": np.random.choice(["FF", "SL", "CH", "CU"], n),
        "delta_run_exp": np.random.normal(0, 0.15, n),
    })


class TestPlatoonAdvantage:
    def test_same_hand_is_advantage(self, sample_df):
        df = add_platoon_advantage(sample_df)
        # RHP vs RHB should be 1
        mask = (df["p_throws"] == "R") & (df["stand"] == "R")
        assert (df.loc[mask, "platoon_adv"] == 1).all()

    def test_opposite_hand_no_advantage(self, sample_df):
        df = add_platoon_advantage(sample_df)
        # RHP vs LHB should be 0
        mask = (df["p_throws"] == "R") & (df["stand"] == "L")
        assert (df.loc[mask, "platoon_adv"] == 0).all()

    def test_column_exists(self, sample_df):
        df = add_platoon_advantage(sample_df)
        assert "platoon_adv" in df.columns

    def test_binary_values(self, sample_df):
        df = add_platoon_advantage(sample_df)
        assert set(df["platoon_adv"].unique()).issubset({0, 1})


class TestCountLeverage:
    def test_all_counts_mapped(self, sample_df):
        df = add_count_leverage(sample_df)
        assert df["count_leverage"].notna().all()

    def test_known_values(self, sample_df):
        """0-2 should be pitcher-favorable (positive), 3-0 should be hitter-favorable (negative)."""
        assert COUNT_LEVERAGE[(0, 2)] > 0  # Pitcher ahead
        assert COUNT_LEVERAGE[(3, 0)] < 0  # Hitter ahead

    def test_column_exists(self, sample_df):
        df = add_count_leverage(sample_df)
        assert "count_leverage" in df.columns


class TestStuffComposite:
    def test_column_exists(self, sample_df):
        df = add_stuff_composite(sample_df)
        assert "stuff_composite" in df.columns
        assert "total_movement" in df.columns

    def test_no_nans(self, sample_df):
        df = add_stuff_composite(sample_df)
        assert df["stuff_composite"].notna().all()

    def test_roughly_centered(self, sample_df):
        """Z-scored composite should be roughly mean-zero across full data."""
        df = add_stuff_composite(sample_df)
        assert abs(df["stuff_composite"].mean()) < 0.5  # Generous tolerance


class TestLocationZone:
    def test_all_zones_present(self, sample_df):
        df = add_location_zone(sample_df)
        assert "zone" in df.columns
        zones = set(df["zone"].unique())
        # Should have at least some of these
        assert len(zones.intersection({"heart", "edge", "chase", "waste"})) >= 2

    def test_heart_is_center(self, sample_df):
        """Pitches at (0, 2.5) should be in the heart."""
        df = sample_df.copy()
        df.loc[:, "plate_x"] = 0.0
        df.loc[:, "plate_z"] = 2.5
        df = add_location_zone(df)
        assert (df["zone"] == "heart").all()

    def test_one_hot_columns(self, sample_df):
        df = add_location_zone(sample_df)
        zone_cols = [c for c in df.columns if c.startswith("zone_")]
        assert len(zone_cols) >= 1  # At least one dummy after drop_first


class TestBaseState:
    def test_bases_empty(self, sample_df):
        df = sample_df.copy()
        df["on_1b"] = 0
        df["on_2b"] = 0
        df["on_3b"] = 0
        df = add_base_state(df)
        assert (df["base_state"] == 0).all()

    def test_bases_loaded(self, sample_df):
        df = sample_df.copy()
        df["on_1b"] = 1
        df["on_2b"] = 1
        df["on_3b"] = 1
        df = add_base_state(df)
        assert (df["base_state"] == 7).all()

    def test_range(self, sample_df):
        df = add_base_state(sample_df)
        assert df["base_state"].min() >= 0
        assert df["base_state"].max() <= 7


class TestPitcherIndex:
    def test_contiguous(self, sample_df):
        df, pitcher_map = create_pitcher_index(sample_df)
        indices = sorted(df["pitcher_idx"].unique())
        assert indices == list(range(len(indices)))

    def test_map_size(self, sample_df):
        df, pitcher_map = create_pitcher_index(sample_df)
        assert len(pitcher_map) == sample_df["pitcher"].nunique()

    def test_no_nans(self, sample_df):
        df, _ = create_pitcher_index(sample_df)
        assert df["pitcher_idx"].notna().all()
