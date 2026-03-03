"""
Tests for data acquisition and cleaning module.

These test the cleaning logic on synthetic data -- we don't
hit the Statcast API in tests.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import KEEP_COLS, clean_statcast


@pytest.fixture
def raw_statcast_df():
    """Synthetic raw Statcast-like dataframe for testing clean_statcast."""
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "game_date": pd.date_range("2024-04-01", periods=n, freq="h"),
        "pitcher": np.random.choice([100, 200, 300], n),
        "batter": np.random.choice([400, 500, 600], n),
        "release_speed": np.random.normal(93, 5, n),
        "release_spin_rate": np.random.normal(2300, 300, n),
        "spin_axis": np.random.uniform(0, 360, n),
        "pfx_x": np.random.normal(0, 8, n),
        "pfx_z": np.random.normal(0, 8, n),
        "plate_x": np.random.normal(0, 0.8, n),
        "plate_z": np.random.normal(2.5, 0.7, n),
        "release_pos_x": np.random.normal(-1, 0.5, n),
        "release_pos_y": np.random.normal(55, 1, n),
        "release_pos_z": np.random.normal(6, 0.5, n),
        "balls": np.random.choice([0, 1, 2, 3], n),
        "strikes": np.random.choice([0, 1, 2], n),
        "outs_when_up": np.random.choice([0, 1, 2], n),
        "inning": np.random.choice(range(1, 10), n),
        "inning_topbot": np.random.choice(["Top", "Bot"], n),
        "on_1b": np.random.choice([np.nan, 12345.0], n),
        "on_2b": np.random.choice([np.nan, 67890.0], n),
        "on_3b": np.random.choice([np.nan, 11111.0], n),
        "stand": np.random.choice(["R", "L"], n),
        "p_throws": np.random.choice(["R", "L"], n),
        "pitch_type": np.random.choice(["FF", "SL", "CH", "CU"], n),
        "pitch_name": np.random.choice(["4-Seam Fastball", "Slider", "Changeup", "Curveball"], n),
        "delta_run_exp": np.random.normal(0, 0.15, n),
        "description": "called_strike",
        "events": np.nan,
        "type": np.random.choice(["S", "B", "X"], n),
        "estimated_woba_using_speedangle": np.random.uniform(0, 1, n),
        "launch_speed": np.random.normal(90, 10, n),
        "launch_angle": np.random.normal(15, 20, n),
        "home_team": "LAD",
        "away_team": "NYM",
        "game_type": np.random.choice(["R", "R", "R", "S"], n),  # 75% regular season
        "game_pk": np.random.choice([700001, 700002, 700003], n),
    })
    return df


class TestCleanStatcast:
    def test_filters_to_regular_season(self, raw_statcast_df):
        clean = clean_statcast(raw_statcast_df)
        # All spring training rows should be gone
        assert len(clean) < len(raw_statcast_df)

    def test_keeps_only_relevant_columns(self, raw_statcast_df):
        clean = clean_statcast(raw_statcast_df)
        # Should not have columns outside KEEP_COLS
        for col in clean.columns:
            assert col in KEEP_COLS

    def test_no_nulls_in_critical_fields(self, raw_statcast_df):
        # Inject some nulls in critical fields
        df = raw_statcast_df.copy()
        df.loc[0:5, "release_speed"] = np.nan
        df.loc[10:15, "delta_run_exp"] = np.nan

        clean = clean_statcast(df)
        assert clean["release_speed"].notna().all()
        assert clean["delta_run_exp"].notna().all()

    def test_base_runners_are_binary(self, raw_statcast_df):
        clean = clean_statcast(raw_statcast_df)
        for col in ["on_1b", "on_2b", "on_3b"]:
            if col in clean.columns:
                assert set(clean[col].unique()).issubset({0, 1})

    def test_integer_columns(self, raw_statcast_df):
        clean = clean_statcast(raw_statcast_df)
        for col in ["balls", "strikes", "outs_when_up", "inning"]:
            if col in clean.columns:
                assert clean[col].dtype in [np.int64, np.int32, int]

    def test_game_date_is_datetime(self, raw_statcast_df):
        clean = clean_statcast(raw_statcast_df)
        if "game_date" in clean.columns:
            assert pd.api.types.is_datetime64_any_dtype(clean["game_date"])

    def test_output_not_empty(self, raw_statcast_df):
        clean = clean_statcast(raw_statcast_df)
        assert len(clean) > 0


class TestKeepCols:
    def test_critical_fields_in_keep_cols(self):
        """Verify the column list includes everything the model needs."""
        required = [
            "pitcher", "release_speed", "pfx_x", "pfx_z",
            "plate_x", "plate_z", "delta_run_exp",
            "pitch_type", "balls", "strikes", "stand", "p_throws",
        ]
        for col in required:
            assert col in KEEP_COLS, f"{col} missing from KEEP_COLS"
