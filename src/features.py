"""
Feature engineering for the Bayesian pitch model.

Transforms raw Statcast columns into model-ready features:
- Platoon advantage encoding
- Stuff+ composite metric
- Count leverage states
- Location zone classification
- Pitcher/batter index mapping for hierarchical model
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- Count leverage ---
# Not all counts are equal. 0-2 is pitcher dominant, 3-0 is hitter dominant.
# We encode this as a single numeric feature.
COUNT_LEVERAGE = {
    (0, 0): 0.0,
    (1, 0): -0.1,
    (0, 1): 0.15,
    (2, 0): -0.25,
    (1, 1): 0.0,
    (0, 2): 0.35,
    (3, 0): -0.45,
    (2, 1): -0.1,
    (1, 2): 0.2,
    (3, 1): -0.3,
    (2, 2): 0.05,
    (3, 2): -0.1,
}


def add_platoon_advantage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flag: 1 if pitcher has platoon advantage, 0 otherwise.

    Platoon advantage = same-side matchup for pitcher:
    - RHP vs RHB = advantage (1)
    - LHP vs LHB = advantage (1)
    - RHP vs LHB = no advantage (0)
    - LHP vs RHB = no advantage (0)
    """
    df = df.copy()
    df["platoon_adv"] = (df["p_throws"] == df["stand"]).astype(int)
    return df


def add_count_leverage(df: pd.DataFrame) -> pd.DataFrame:
    """Add count leverage score based on ball-strike state."""
    df = df.copy()
    df["count_leverage"] = df.apply(
        lambda r: COUNT_LEVERAGE.get((r["balls"], r["strikes"]), 0.0),
        axis=1,
    )
    return df


def add_stuff_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stuff composite: z-scored combination of velocity + total movement.

    Higher = nastier pitch. Simple version of Stuff+ without the
    full model -- good enough for a feature, and you can explain
    the math in the notebook.
    """
    df = df.copy()

    # Total movement magnitude
    df["total_movement"] = np.sqrt(df["pfx_x"] ** 2 + df["pfx_z"] ** 2)

    # Z-score within pitch type (a 95 mph fastball and 85 mph slider
    # shouldn't be on the same scale)
    for col in ["release_speed", "total_movement"]:
        grouped = df.groupby("pitch_type")[col]
        df[f"{col}_z"] = grouped.transform(lambda x: (x - x.mean()) / x.std().clip(min=0.01))

    # Composite: equal weight velocity + movement
    df["stuff_composite"] = (df["release_speed_z"] + df["total_movement_z"]) / 2

    # Clean up intermediate columns
    df = df.drop(columns=["release_speed_z", "total_movement_z"])

    return df


def add_location_zone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify pitch location into zones.

    Uses plate_x and plate_z to create:
    - zone_heart: center of strike zone
    - zone_edge: edges of strike zone
    - zone_chase: just outside (where you get swings on bad pitches)
    - zone_waste: way outside

    Strike zone is roughly plate_x in [-0.83, 0.83] feet,
    plate_z in [1.5, 3.5] feet (varies by batter but this is standard).
    """
    df = df.copy()

    x, z = df["plate_x"].abs(), df["plate_z"]

    # Vertical in zone
    z_in = (z >= 1.5) & (z <= 3.5)
    z_near = (z >= 1.2) & (z <= 3.8)

    # Horizontal in zone
    x_in = x <= 0.83
    x_near = x <= 1.1

    heart = x_in & (x <= 0.5) & z_in & (z >= 1.8) & (z <= 3.2)
    edge = (x_in & z_in) & ~heart
    chase = (x_near & z_near) & ~(x_in & z_in)

    df["zone"] = "waste"
    df.loc[chase, "zone"] = "chase"
    df.loc[edge, "zone"] = "edge"
    df.loc[heart, "zone"] = "heart"

    # One-hot encode for the model
    zone_dummies = pd.get_dummies(df["zone"], prefix="zone", drop_first=True)
    df = pd.concat([df, zone_dummies], axis=1)

    return df


def add_base_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode base-runner state as a single integer 0-7.

    000 = bases empty, 111 = bases loaded.
    This captures the full base state in one feature.
    """
    df = df.copy()
    df["base_state"] = df["on_1b"] * 1 + df["on_2b"] * 2 + df["on_3b"] * 4
    return df


def create_pitcher_index(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Map pitcher IDs to contiguous 0-indexed integers for PyMC.

    Returns:
        df with 'pitcher_idx' column
        pitcher_map: dict mapping pitcher_id -> index
    """
    df = df.copy()
    unique_pitchers = df["pitcher"].unique()
    pitcher_map = {pid: idx for idx, pid in enumerate(sorted(unique_pitchers))}
    df["pitcher_idx"] = df["pitcher"].map(pitcher_map)
    return df, pitcher_map


def build_model_matrix(
    df: pd.DataFrame,
    scale: bool = True,
) -> tuple[pd.DataFrame, StandardScaler | None]:
    """
    Run the full feature pipeline and return model-ready data.

    Args:
        df: Cleaned Statcast dataframe
        scale: Whether to standardize continuous features

    Returns:
        df: Feature-enriched dataframe with pitcher_idx
        scaler: Fitted StandardScaler (None if scale=False)
    """
    # Feature engineering pipeline
    df = add_platoon_advantage(df)
    df = add_count_leverage(df)
    df = add_stuff_composite(df)
    df = add_location_zone(df)
    df = add_base_state(df)
    df, pitcher_map = create_pitcher_index(df)

    # Continuous features to standardize
    continuous_cols = [
        "release_speed", "release_spin_rate",
        "pfx_x", "pfx_z",
        "plate_x", "plate_z",
        "stuff_composite", "total_movement",
        "count_leverage",
    ]
    continuous_cols = [c for c in continuous_cols if c in df.columns]

    # Drop rows with NaN in any model-critical column before scaling.
    # Feature engineering can introduce NaNs (e.g., pitch type groups
    # with n=1 produce NaN std in stuff composite z-scoring).
    model_cols = continuous_cols + ["delta_run_exp", "platoon_adv", "pitcher_idx"]
    model_cols = [c for c in model_cols if c in df.columns]
    before = len(df)
    df = df.dropna(subset=model_cols).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        import logging as _log
        _log.getLogger(__name__).info("Dropped %d rows with NaN in model columns", dropped)

    scaler = None
    if scale:
        scaler = StandardScaler()
        df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    # Store pitcher map as attribute for downstream access
    df.attrs["pitcher_map"] = pitcher_map
    df.attrs["n_pitchers"] = len(pitcher_map)
    df.attrs["continuous_cols"] = continuous_cols

    return df, scaler
