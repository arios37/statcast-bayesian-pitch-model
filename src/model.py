"""
Bayesian hierarchical model for pitch-level expected run value.

Uses PyMC to build a hierarchical regression with:
- Pitcher-level random intercepts (partial pooling)
- Fixed effects for pitch physics, location, count, matchup
- Normal likelihood on delta_run_exp

The partial pooling is the key insight: low-sample pitchers get
shrunk toward the league mean, high-sample pitchers express their
true tendencies. This is textbook hierarchical Bayesian modeling.
"""

from __future__ import annotations

import logging
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

logger = logging.getLogger(__name__)


def build_model(
    df: pd.DataFrame,
    sample: bool = False,
    sample_kwargs: dict[str, Any] | None = None,
) -> tuple[pm.Model, az.InferenceData | None]:
    """
    Build and optionally sample the hierarchical pitch model.

    Args:
        df: Feature-engineered dataframe from features.build_model_matrix()
            Must contain pitcher_idx and all feature columns.
        sample: Whether to run MCMC sampling
        sample_kwargs: Keyword args passed to pm.sample()

    Returns:
        model: Compiled PyMC model
        idata: ArviZ InferenceData if sampled, else None
    """
    if sample_kwargs is None:
        sample_kwargs = {}

    n_pitchers = df["pitcher_idx"].nunique()
    pitcher_idx = df["pitcher_idx"].values

    # Extract feature arrays
    velo = df["release_speed"].values
    spin = df["release_spin_rate"].values if "release_spin_rate" in df.columns else None
    hmov = df["pfx_x"].values
    vmov = df["pfx_z"].values
    loc_x = df["plate_x"].values
    loc_z = df["plate_z"].values
    count_lev = df["count_leverage"].values
    platoon = df["platoon_adv"].values
    stuff = df["stuff_composite"].values if "stuff_composite" in df.columns else None

    # Target
    y_obs = df["delta_run_exp"].values

    # Sanity check: NaN in features or target will silently poison the sampler
    _arrays = {"velo": velo, "hmov": hmov, "vmov": vmov, "loc_x": loc_x,
               "loc_z": loc_z, "count_lev": count_lev, "platoon": platoon,
               "y_obs": y_obs}
    if stuff is not None:
        _arrays["stuff"] = stuff
    if spin is not None:
        _arrays["spin"] = spin
    for name, arr in _arrays.items():
        n_nan = int(np.isnan(arr).sum())
        if n_nan > 0:
            msg = f"Found {n_nan} NaN values in '{name}'. Clean your data before modeling."
            raise ValueError(msg)

    logger.info(
        "Building model: %s pitches, %s pitchers",
        f"{len(df):,}",
        f"{n_pitchers:,}",
    )

    coords = {"pitcher": np.arange(n_pitchers)}

    with pm.Model(coords=coords) as model:
        # === Data containers ===
        pitcher_idx_data = pm.Data("pitcher_idx", pitcher_idx)

        # === Hyperpriors (league-level parameters) ===
        # These control the distribution FROM WHICH pitcher effects are drawn.
        # mu_alpha = league-average baseline run value tendency
        # sigma_alpha = how much pitchers vary from that average
        mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=0.1)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.05)

        # === Pitcher-level random intercepts ===
        # Each pitcher gets their own intercept, drawn from the league distribution.
        # This is the hierarchical part: partial pooling.
        # Non-centered parameterization for better sampling geometry.
        alpha_pitcher_offset = pm.Normal(
            "alpha_pitcher_offset", mu=0, sigma=1, dims="pitcher"
        )
        alpha_pitcher = pm.Deterministic(
            "alpha_pitcher",
            mu_alpha + sigma_alpha * alpha_pitcher_offset,
            dims="pitcher",
        )

        # === Fixed effects (population-level) ===
        beta_velo = pm.Normal("beta_velo", mu=0, sigma=0.1)
        beta_hmov = pm.Normal("beta_hmov", mu=0, sigma=0.1)
        beta_vmov = pm.Normal("beta_vmov", mu=0, sigma=0.1)
        beta_loc_x = pm.Normal("beta_loc_x", mu=0, sigma=0.1)
        beta_loc_z = pm.Normal("beta_loc_z", mu=0, sigma=0.1)
        beta_count = pm.Normal("beta_count", mu=0, sigma=0.1)
        beta_platoon = pm.Normal("beta_platoon", mu=0, sigma=0.1)

        # Optional: stuff composite and spin
        if stuff is not None:
            beta_stuff = pm.Normal("beta_stuff", mu=0, sigma=0.1)
        if spin is not None:
            beta_spin = pm.Normal("beta_spin", mu=0, sigma=0.1)

        # === Linear predictor ===
        mu = (
            alpha_pitcher[pitcher_idx_data]
            + beta_velo * velo
            + beta_hmov * hmov
            + beta_vmov * vmov
            + beta_loc_x * loc_x
            + beta_loc_z * loc_z
            + beta_count * count_lev
            + beta_platoon * platoon
        )

        if stuff is not None:
            mu = mu + beta_stuff * stuff
        if spin is not None:
            mu = mu + beta_spin * spin

        # === Likelihood ===
        sigma = pm.HalfNormal("sigma", sigma=0.1)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_obs)

    logger.info("Model built: %s free parameters", len(model.free_RVs))

    # === Sample ===
    idata = None
    if sample:
        defaults = {
            "draws": 2000,
            "tune": 1000,
            "cores": 4,
            "target_accept": 0.9,
            "return_inferencedata": True,
            "random_seed": 42,
        }
        defaults.update(sample_kwargs)
        logger.info("Sampling with %s", defaults)

        with model:
            idata = pm.sample(**defaults)
            # Add posterior predictive for model checking
            idata.extend(pm.sample_posterior_predictive(idata))

    return model, idata


def build_model_subsample(
    df: pd.DataFrame,
    n_pitches: int = 50_000,
    n_pitchers_keep: int | None = None,
    sample_kwargs: dict[str, Any] | None = None,
) -> tuple[pm.Model, az.InferenceData]:
    """
    Build and sample on a subsample for faster iteration.

    Full 700K pitches takes a while. This lets you prototype
    on a smaller slice, then scale up when the model looks right.

    Args:
        df: Full feature-engineered dataframe
        n_pitches: Number of pitches to sample
        n_pitchers_keep: If set, keep only top-N pitchers by pitch count
        sample_kwargs: Passed to pm.sample()
    """
    sub = df.copy()

    # Optionally filter to high-volume pitchers first
    if n_pitchers_keep is not None:
        top_pitchers = (
            sub.groupby("pitcher")
            .size()
            .nlargest(n_pitchers_keep)
            .index
        )
        sub = sub[sub["pitcher"].isin(top_pitchers)]

        # Re-index pitchers to be contiguous
        unique = sub["pitcher"].unique()
        remap = {pid: idx for idx, pid in enumerate(sorted(unique))}
        sub["pitcher_idx"] = sub["pitcher"].map(remap)

    # Random subsample of pitches
    if len(sub) > n_pitches:
        sub = sub.sample(n=n_pitches, random_state=42)

    return build_model(sub, sample=True, sample_kwargs=sample_kwargs)


def get_diagnostics(idata: az.InferenceData) -> pd.DataFrame:
    """
    Extract key MCMC diagnostics: R-hat, ESS, divergences.

    R-hat should be < 1.01 for all parameters.
    ESS (effective sample size) should be > 400.
    Divergences should be 0.
    """
    summary = az.summary(
        idata,
        var_names=["mu_alpha", "sigma_alpha", "beta_velo", "beta_hmov",
                    "beta_vmov", "beta_loc_x", "beta_loc_z",
                    "beta_count", "beta_platoon", "sigma"],
        round_to=4,
    )

    # Check for divergences
    if hasattr(idata, "sample_stats"):
        n_div = int(idata.sample_stats.diverging.sum().values)
        logger.info("Divergences: %d", n_div)
        if n_div > 0:
            logger.warning(
                "Model has %d divergences. Consider reparameterizing "
                "or increasing target_accept.",
                n_div,
            )

    return summary


def get_pitcher_effects(
    idata: az.InferenceData,
    pitcher_map: dict[int, int] | None = None,
) -> pd.DataFrame:
    """
    Extract posterior summary for pitcher-level random effects.

    Returns dataframe with mean, sd, and 94% HDI for each pitcher.
    Sorted by posterior mean (best to worst).
    """
    summary = az.summary(
        idata,
        var_names=["alpha_pitcher"],
        hdi_prob=0.94,
    )

    if pitcher_map is not None:
        # Reverse the map: index -> pitcher_id
        reverse_map = {v: k for k, v in pitcher_map.items()}
        summary["pitcher_id"] = [reverse_map.get(i, i) for i in range(len(summary))]

    return summary.sort_values("mean")
