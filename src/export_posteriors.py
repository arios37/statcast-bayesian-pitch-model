"""
Export posterior summaries and pitch-level data to JSON for the React frontend.

Reads the cached InferenceData (idata_subsample.nc) and raw Statcast data,
then builds a JSON payload consumed by the Bayesian Pitch Model Explorer.

Two paths for pitcher-level posteriors:
1. If the pitcher is IN the subsample model: use their actual alpha_pitcher posterior.
2. If the pitcher is NOT in the model (most Dodgers): use the estimated hyperprior
   (mu_alpha, sigma_alpha) as a prior, then do a conjugate normal-normal update
   with their raw pitch-level delta_run_exp data. This gives a proper Bayesian
   posterior without re-running MCMC.

Usage:
    python -m src.export_posteriors                         # defaults
    python -m src.export_posteriors --year 2025 --output frontend/public/data/posteriors.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from src.data import DATA_DIR, get_data
from src.features import build_model_matrix

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# 2026 Dodgers pitching staff
DODGERS_PITCHERS = {
    "Yoshinobu Yamamoto": 808967,
    "Tyler Glasnow": 607192,
    "Shohei Ohtani": 660271,
    "Blake Snell": 605483,
    "Roki Sasaki": 808963,
    "Emmet Sheehan": 686218,
    "Edwin Diaz": 621242,
    "Tanner Scott": 656945,
    "Blake Treinen": 595014,
    "Alex Vesia": 681911,
    "Evan Phillips": 623465,
    "Gavin Stone": 694813,
    "Edgardo Henriquez": 683618,
}

PITCH_NAMES = {
    "FF": "4-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "CU": "Curveball",
    "CH": "Changeup",
    "FS": "Splitter",
    "ST": "Sweeper",
    "KC": "Knuckle Curve",
    "SV": "Screwball",
}

PITCH_COLORS = {
    "FF": "#e74c3c",
    "SI": "#f39c12",
    "FC": "#9b59b6",
    "SL": "#3498db",
    "CU": "#2ecc71",
    "CH": "#e67e22",
    "FS": "#264653",
    "ST": "#e76f51",
    "KC": "#6a4c93",
    "SV": "#a8dadc",
}


def load_idata(path: Path | None = None) -> az.InferenceData:
    """Load cached InferenceData from netcdf."""
    if path is None:
        path = DATA_DIR / "idata_subsample.nc"
    if not path.exists():
        raise FileNotFoundError(
            f"No InferenceData at {path}. Run notebook 03 or 04 first."
        )
    idata = az.from_netcdf(str(path))
    logger.info("Loaded InferenceData from %s", path)
    return idata


def extract_hyperprior(idata: az.InferenceData) -> dict:
    """
    Extract the hyperprior posterior samples: mu_alpha and sigma_alpha.

    These define the league-level distribution from which pitcher effects are drawn.
    We use the posterior means as point estimates for the conjugate update.
    """
    mu_alpha_samples = idata.posterior["mu_alpha"].values.flatten()
    sigma_alpha_samples = idata.posterior["sigma_alpha"].values.flatten()

    return {
        "mu_alpha_mean": float(np.mean(mu_alpha_samples)),
        "mu_alpha_sd": float(np.std(mu_alpha_samples)),
        "sigma_alpha_mean": float(np.mean(sigma_alpha_samples)),
        "sigma_alpha_sd": float(np.std(sigma_alpha_samples)),
        "sigma_obs": float(np.mean(idata.posterior["sigma"].values.flatten())),
    }


def conjugate_normal_update(
    prior_mu: float,
    prior_sigma: float,
    data_mean: float,
    data_sigma: float,
    n: int,
) -> tuple[float, float]:
    """
    Conjugate normal-normal posterior update.

    Given:
        prior: N(prior_mu, prior_sigma^2)
        likelihood: y_bar ~ N(theta, data_sigma^2 / n)

    Posterior:
        theta | y ~ N(posterior_mu, posterior_sigma^2)

    This is the exact same thing PyMC does for the hierarchical model,
    just computed analytically instead of via MCMC.
    """
    prior_prec = 1 / prior_sigma**2
    data_prec = n / data_sigma**2

    posterior_prec = prior_prec + data_prec
    posterior_sigma = np.sqrt(1 / posterior_prec)
    posterior_mu = (prior_prec * prior_mu + data_prec * data_mean) / posterior_prec

    return float(posterior_mu), float(posterior_sigma)


def build_pitcher_posterior(
    name: str,
    pitcher_id: int,
    df_raw: pd.DataFrame,
    hyperprior: dict,
    idata: az.InferenceData,
    pitcher_map: dict | None,
) -> dict | None:
    """
    Build posterior summary for a single pitcher.

    If the pitcher is in the model, use their MCMC posterior.
    Otherwise, use the conjugate update with the hyperprior.
    """
    df_pitcher = df_raw[df_raw["pitcher"] == pitcher_id]
    if len(df_pitcher) < 10:
        logger.warning("Skipping %s: only %d pitches", name, len(df_pitcher))
        return None

    n_pitches = len(df_pitcher)
    raw_mean_dre = float(df_pitcher["delta_run_exp"].mean())
    raw_sd_dre = float(df_pitcher["delta_run_exp"].std())  # noqa: F841

    # Check if pitcher is in the model
    in_model = False
    if pitcher_map and pitcher_id in pitcher_map:
        idx = pitcher_map[pitcher_id]
        alpha_samples = idata.posterior["alpha_pitcher"].values[:, :, idx].flatten()
        posterior_mean = float(np.mean(alpha_samples))
        posterior_sd = float(np.std(alpha_samples))
        in_model = True
        logger.info("%s: in model (idx=%d), posterior mean=%.4f", name, idx, posterior_mean)
    else:
        # Conjugate update using the hyperprior
        posterior_mean, posterior_sd = conjugate_normal_update(
            prior_mu=hyperprior["mu_alpha_mean"],
            prior_sigma=hyperprior["sigma_alpha_mean"],
            data_mean=raw_mean_dre,
            data_sigma=hyperprior["sigma_obs"],
            n=n_pitches,
        )
        logger.info(
            "%s: conjugate update (n=%d), raw=%.4f, posterior=%.4f",
            name, n_pitches, raw_mean_dre, posterior_mean,
        )

    # Pitch-level arsenal breakdown
    arsenal = {}
    for pitch_type, group in df_pitcher.groupby("pitch_type"):
        if len(group) < 5:
            continue
        usage = len(group) / n_pitches
        arsenal[pitch_type] = {
            "name": PITCH_NAMES.get(pitch_type, pitch_type),
            "color": PITCH_COLORS.get(pitch_type, "#888888"),
            "velo": round(float(group["release_speed"].mean()), 1),
            "spin": round(float(group["release_spin_rate"].mean()), 0),
            "pfx_x": round(float(group["pfx_x"].mean()), 1),
            "pfx_z": round(float(group["pfx_z"].mean()), 1),
            "usage": round(usage, 3),
            "rawDRE": round(float(group["delta_run_exp"].mean()), 4),
        }

    # Estimate ERA from delta_run_exp (rough approximation)
    # dRE per pitch * pitches per IP (~15) * 9 innings + league avg ERA
    pitches_per_ip = 15.5
    estimated_era = max(0, 4.25 + raw_mean_dre * pitches_per_ip * 9)
    estimated_ip = n_pitches / pitches_per_ip

    return {
        "id": name.lower().replace(" ", "_"),
        "name": name,
        "pitcherId": pitcher_id,
        "inModel": in_model,
        "pitches": arsenal,
        "nPitches": n_pitches,
        "rawMeanDRE": round(raw_mean_dre, 6),
        "posteriorMean": round(posterior_mean, 6),
        "posteriorSD": round(posterior_sd, 6),
        "ci90": [
            round(posterior_mean - 1.645 * posterior_sd, 6),
            round(posterior_mean + 1.645 * posterior_sd, 6),
        ],
        "ci94": [
            round(posterior_mean - 1.88 * posterior_sd, 6),
            round(posterior_mean + 1.88 * posterior_sd, 6),
        ],
        "estimatedERA": round(estimated_era, 2),
        "estimatedIP": round(estimated_ip, 1),
    }


def extract_fixed_effects(idata: az.InferenceData) -> dict:
    """Extract posterior summaries for all fixed-effect betas."""
    beta_vars = [
        "beta_velo", "beta_hmov", "beta_vmov",
        "beta_loc_x", "beta_loc_z",
        "beta_count", "beta_platoon",
    ]
    # Include optional vars if present
    for v in ["beta_stuff", "beta_spin"]:
        if v in idata.posterior:
            beta_vars.append(v)

    effects = {}
    for var in beta_vars:
        if var not in idata.posterior:
            continue
        samples = idata.posterior[var].values.flatten()
        effects[var] = {
            "mean": round(float(np.mean(samples)), 6),
            "sd": round(float(np.std(samples)), 6),
            "hdi94": [
                round(float(np.percentile(samples, 3)), 6),
                round(float(np.percentile(samples, 97)), 6),
            ],
        }

    return effects


def extract_diagnostics(idata: az.InferenceData) -> dict:
    """Extract MCMC diagnostics summary."""
    n_div = 0
    if hasattr(idata, "sample_stats"):
        n_div = int(idata.sample_stats.diverging.sum().values)

    summary = az.summary(
        idata,
        var_names=["mu_alpha", "sigma_alpha", "sigma"],
        round_to=6,
    )

    return {
        "divergences": n_div,
        "parameters": {
            row_name: {
                "mean": round(float(row["mean"]), 6),
                "sd": round(float(row["sd"]), 6),
                "r_hat": round(float(row["r_hat"]), 4),
                "ess_bulk": round(float(row["ess_bulk"]), 0),
            }
            for row_name, row in summary.iterrows()
        },
    }


def build_export(
    year: int = 2025,
    idata_path: Path | None = None,
) -> dict:
    """
    Build the full JSON export for the React frontend.

    Returns a dict ready for json.dump().
    """
    # Load data
    df_raw = get_data(year=year)
    df, scaler = build_model_matrix(df_raw)
    pitcher_map = df.attrs.get("pitcher_map", {})

    # Load model
    idata = load_idata(idata_path)
    hyperprior = extract_hyperprior(idata)

    logger.info("Hyperprior: mu_alpha=%.4f, sigma_alpha=%.4f",
                hyperprior["mu_alpha_mean"], hyperprior["sigma_alpha_mean"])

    # Build pitcher posteriors
    pitchers = {}
    for name, pid in DODGERS_PITCHERS.items():
        result = build_pitcher_posterior(
            name, pid, df_raw, hyperprior, idata, pitcher_map,
        )
        if result is not None:
            pitchers[name] = result

    # Fixed effects
    fixed_effects = extract_fixed_effects(idata)

    # Diagnostics
    diagnostics = extract_diagnostics(idata)

    # League-level stats
    league_mean_dre = float(df_raw["delta_run_exp"].mean())

    return {
        "_meta": {
            "generated": pd.Timestamp.now().isoformat(),
            "year": year,
            "totalPitches": len(df_raw),
            "totalPitchers": df_raw["pitcher"].nunique(),
            "modelType": "hierarchical_normal",
            "samplerInfo": {
                "draws": 1000,
                "tune": 500,
                "chains": 4,
                "targetAccept": 0.9,
                "subsampleSize": 50000,
                "nPitchersInModel": 100,
            },
        },
        "hyperprior": hyperprior,
        "leagueMeanDRE": round(league_mean_dre, 6),
        "fixedEffects": fixed_effects,
        "diagnostics": diagnostics,
        "pitchers": pitchers,
    }


def main():
    parser = argparse.ArgumentParser(description="Export posterior data to JSON")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--idata", type=str, default=None, help="Path to idata .nc file")
    parser.add_argument(
        "--output", type=str,
        default="frontend/public/data/posteriors.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    idata_path = Path(args.idata) if args.idata else None
    export = build_export(year=args.year, idata_path=idata_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(export, f, indent=2)

    n_pitchers = len(export["pitchers"])
    logger.info("Exported %d pitchers to %s", n_pitchers, output_path)
    print(f"\nExported {n_pitchers} Dodgers pitchers to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
