"""
Data acquisition and cleaning for Statcast pitch-level data.

Pulls from Baseball Savant via pybaseball, cleans nulls,
filters to regular season, and stores as parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pybaseball import cache, statcast

logger = logging.getLogger(__name__)

# Enable pybaseball cache so repeat pulls don't hammer the API
cache.enable()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Columns we actually need -- drop the noise upfront
KEEP_COLS = [
    # identifiers
    "game_date", "pitcher", "batter", "game_pk",
    # pitch physics
    "release_speed", "release_spin_rate", "spin_axis",
    "pfx_x", "pfx_z",
    "plate_x", "plate_z",
    "release_pos_x", "release_pos_y", "release_pos_z",
    # game state
    "balls", "strikes", "outs_when_up",
    "inning", "inning_topbot",
    "on_1b", "on_2b", "on_3b",
    # matchup
    "stand", "p_throws", "pitch_type", "pitch_name",
    # outcome
    "delta_run_exp", "description", "events", "type",
    "estimated_woba_using_speedangle",
    "launch_speed", "launch_angle",
    # game context
    "home_team", "away_team",
    "game_type",
]


def pull_statcast_season(
    year: int = 2025,
    start_dt: str | None = None,
    end_dt: str | None = None,
) -> pd.DataFrame:
    """
    Pull a full season of Statcast data in ~2-week chunks.

    pybaseball caps single queries at ~40 days, so we chunk it.
    Returns raw dataframe before any cleaning.
    """
    if start_dt is None:
        start_dt = f"{year}-03-20"  # spring training cutoff
    if end_dt is None:
        end_dt = f"{year}-10-01"  # regular season end

    logger.info("Pulling Statcast data: %s to %s", start_dt, end_dt)

    # Pull in 14-day windows to stay under API limits
    chunks = []
    current = pd.Timestamp(start_dt)
    end = pd.Timestamp(end_dt)

    while current < end:
        chunk_end = min(current + pd.Timedelta(days=13), end)
        logger.info("  Chunk: %s to %s", current.date(), chunk_end.date())
        try:
            chunk = statcast(
                start_dt=str(current.date()),
                end_dt=str(chunk_end.date()),
            )
            if chunk is not None and len(chunk) > 0:
                chunks.append(chunk)
        except Exception as e:
            logger.warning("Failed chunk %s-%s: %s", current.date(), chunk_end.date(), e)
        current = chunk_end + pd.Timedelta(days=1)

    if not chunks:
        raise RuntimeError("No data returned from Statcast API")

    df = pd.concat(chunks, ignore_index=True)
    logger.info("Raw pull: %s pitches", f"{len(df):,}")
    return df


def clean_statcast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Statcast data:
    1. Filter to regular season only
    2. Keep only columns we need
    3. Drop rows missing critical fields
    4. Type conversions
    """
    n_raw = len(df)

    # 1. Regular season only
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"].copy()
        logger.info("After regular season filter: %s pitches", f"{len(df):,}")

    # 2. Keep relevant columns (intersect with what's actually present)
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()

    # 3. Drop rows where core pitch physics or outcome is missing
    critical_cols = [
        "release_speed", "pfx_x", "pfx_z",
        "plate_x", "plate_z", "delta_run_exp",
        "pitch_type", "balls", "strikes",
    ]
    critical_available = [c for c in critical_cols if c in df.columns]
    before = len(df)
    df = df.dropna(subset=critical_available)
    logger.info("Dropped %s rows with missing critical fields", f"{before - len(df):,}")

    # 4. Type conversions
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])

    for col in ["balls", "strikes", "outs_when_up", "inning"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Encode pitcher/batter as categorical integers for modeling
    if "pitcher" in df.columns:
        df["pitcher"] = df["pitcher"].astype(int)
    if "batter" in df.columns:
        df["batter"] = df["batter"].astype(int)

    # Convert base runners to binary occupied flags
    for base_col in ["on_1b", "on_2b", "on_3b"]:
        if base_col in df.columns:
            df[base_col] = df[base_col].notna().astype(int)

    logger.info("Clean data: %s pitches (kept %.1f%%)", f"{len(df):,}", 100 * len(df) / n_raw)
    return df


def save_parquet(df: pd.DataFrame, filename: str = "statcast_2025.parquet") -> Path:
    """Save cleaned data as parquet for fast reloading."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    df.to_parquet(path, index=False)
    logger.info("Saved to %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def load_parquet(filename: str = "statcast_2025.parquet") -> pd.DataFrame:
    """Load previously saved parquet file."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No data file at {path}. Run pull_statcast_season() first.")
    df = pd.read_parquet(path)
    logger.info("Loaded %s pitches from %s", f"{len(df):,}", path)
    return df


def get_data(year: int = 2025, force_refresh: bool = False) -> pd.DataFrame:
    """
    Main entry point: load from cache or pull fresh.

    This is what the notebooks should call.
    """
    filename = f"statcast_{year}.parquet"
    cache_path = DATA_DIR / filename

    if cache_path.exists() and not force_refresh:
        logger.info("Loading cached data")
        return load_parquet(filename)

    logger.info("No cache found, pulling from Statcast API...")
    raw = pull_statcast_season(year)
    clean = clean_statcast(raw)
    save_parquet(clean, filename)
    return clean
