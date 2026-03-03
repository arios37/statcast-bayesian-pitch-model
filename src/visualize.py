"""
Visualization module for Bayesian pitch model.

Generates:
- Pitch heatmaps (plate location density)
- Movement profile plots by pitch type
- Posterior distribution plots
- Forest plots for pitcher effects
- Posterior predictive checks
- Count leverage charts
"""

from __future__ import annotations

from collections.abc import Sequence

import arviz as az
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Style config
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "FF": "#E63946",    # 4-seam fastball -- red
    "SI": "#F4845F",    # sinker -- orange
    "FC": "#F7B267",    # cutter -- gold
    "SL": "#457B9D",    # slider -- blue
    "CU": "#1D3557",    # curveball -- navy
    "CH": "#2A9D8F",    # changeup -- teal
    "FS": "#264653",    # splitter -- dark teal
    "KC": "#6A4C93",    # knuckle curve -- purple
    "ST": "#E76F51",    # sweeper -- coral
    "SV": "#A8DADC",    # screwball -- light blue
}
DEFAULT_COLOR = "#888888"


def _add_strike_zone(ax: plt.Axes, lw: float = 2) -> None:
    """Draw strike zone rectangle on a pitch location plot."""
    zone = patches.Rectangle(
        (-0.83, 1.5), 1.66, 2.0,
        linewidth=lw, edgecolor="black", facecolor="none", zorder=5,
    )
    ax.add_patch(zone)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 5)
    ax.set_xlabel("Plate X (ft)", fontsize=11)
    ax.set_ylabel("Plate Z (ft)", fontsize=11)
    ax.set_aspect("equal")


def pitch_heatmap(
    df: pd.DataFrame,
    pitch_type: str | None = None,
    pitcher_id: int | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Figure:
    """
    2D kernel density estimate of pitch locations.

    Catcher's perspective (standard baseball convention).
    """
    sub = df.copy()
    if pitch_type:
        sub = sub[sub["pitch_type"] == pitch_type]
    if pitcher_id is not None:
        sub = sub[sub["pitcher"] == pitcher_id]

    if len(sub) < 10:
        raise ValueError(f"Only {len(sub)} pitches match filter. Need >= 10.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 7))
    else:
        fig = ax.figure

    sns.kdeplot(
        x=sub["plate_x"], y=sub["plate_z"],
        fill=True, cmap="YlOrRd", levels=15, thresh=0.05,
        ax=ax,
    )
    _add_strike_zone(ax)

    if title is None:
        parts = []
        if pitch_type:
            parts.append(pitch_type)
        if pitcher_id:
            parts.append(f"Pitcher {pitcher_id}")
        title = " -- ".join(parts) if parts else "All Pitches"
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig


def movement_profile(
    df: pd.DataFrame,
    pitch_types: Sequence[str] | None = None,
    pitcher_id: int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Scatter plot of horizontal vs vertical movement by pitch type.

    This is the classic "pitch movement" plot you see on
    Baseball Savant. pfx_x = horizontal break, pfx_z = induced vertical break.
    """
    sub = df.copy()
    if pitcher_id is not None:
        sub = sub[sub["pitcher"] == pitcher_id]
    if pitch_types:
        sub = sub[sub["pitch_type"].isin(pitch_types)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    for pt, group in sub.groupby("pitch_type"):
        color = COLORS.get(pt, DEFAULT_COLOR)
        ax.scatter(
            group["pfx_x"], group["pfx_z"],
            c=color, alpha=0.15, s=8, label=pt, edgecolors="none",
        )

    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Horizontal Movement (in)", fontsize=12)
    ax.set_ylabel("Induced Vertical Break (in)", fontsize=12)
    ax.set_title("Pitch Movement Profile", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, markerscale=3)

    plt.tight_layout()
    return fig


def posterior_forest_plot(
    idata: az.InferenceData,
    var_names: Sequence[str] | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Forest plot of fixed-effect posterior distributions.

    Shows posterior mean + 94% HDI for each beta coefficient.
    Quick visual: if the interval crosses zero, the effect
    is not clearly different from zero.
    """
    if var_names is None:
        var_names = [
            "beta_velo", "beta_hmov", "beta_vmov",
            "beta_loc_x", "beta_loc_z",
            "beta_count", "beta_platoon",
        ]
        # Only include what exists in the trace
        var_names = [v for v in var_names if v in idata.posterior]

    fig, ax = plt.subplots(figsize=figsize)
    az.plot_forest(
        idata,
        var_names=var_names,
        combined=True,
        hdi_prob=0.94,
        ax=ax,
    )
    ax.axvline(0, color="red", ls="--", lw=1, alpha=0.7)
    ax.set_title("Posterior Distributions of Fixed Effects", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def pitcher_effects_forest(
    idata: az.InferenceData,
    top_n: int = 30,
    figsize: tuple[int, int] = (10, 12),
) -> plt.Figure:
    """
    Forest plot of pitcher-level random effects.

    Shows top-N and bottom-N pitchers by posterior mean.
    This is where you see the partial pooling in action:
    low-sample pitchers get shrunk toward the league mean.
    """
    summary = az.summary(idata, var_names=["alpha_pitcher"], hdi_prob=0.94)
    summary = summary.sort_values("mean")

    # Take top and bottom
    show = pd.concat([summary.head(top_n // 2), summary.tail(top_n // 2)])

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(show))

    ax.errorbar(
        show["mean"], y_pos,
        xerr=[show["mean"] - show["hdi_3%"], show["hdi_97%"] - show["mean"]],
        fmt="o", color="#457B9D", ecolor="#A8DADC", capsize=3, markersize=4,
    )
    ax.axvline(0, color="red", ls="--", lw=1, alpha=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(show.index, fontsize=8)
    ax.set_xlabel("Pitcher Effect (change in expected run value)", fontsize=11)
    ax.set_title(
        f"Pitcher Random Effects (top/bottom {top_n // 2})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    return fig


def posterior_predictive_check(
    idata: az.InferenceData,
    y_obs: np.ndarray,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Posterior predictive check: does the model reproduce the data?

    Overlays observed delta_run_exp distribution against
    samples from the posterior predictive.
    """
    fig, ax = plt.subplots(figsize=figsize)
    az.plot_ppc(idata, observed_rug=False, ax=ax)
    ax.set_xlabel("Delta Run Expectancy", fontsize=11)
    ax.set_title("Posterior Predictive Check", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def trace_diagnostics(
    idata: az.InferenceData,
    var_names: Sequence[str] | None = None,
) -> plt.Figure:
    """
    Trace plots + rank plots for MCMC diagnostics.

    What you're looking for:
    - Trace plots (left): chains should be "fuzzy caterpillars" -- well-mixed
    - Rank plots (right): histograms should be roughly uniform
    """
    if var_names is None:
        var_names = ["mu_alpha", "sigma_alpha", "sigma", "beta_velo", "beta_loc_x"]
        var_names = [v for v in var_names if v in idata.posterior]

    axes = az.plot_trace(idata, var_names=var_names, compact=True)
    fig = axes.ravel()[0].figure
    fig.suptitle("MCMC Trace Diagnostics", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def count_leverage_chart(df: pd.DataFrame, figsize: tuple[int, int] = (10, 5)) -> plt.Figure:
    """
    Bar chart of mean delta_run_exp by count state.

    Shows which counts favor the pitcher vs. the hitter.
    """
    count_means = (
        df.groupby(["balls", "strikes"])["delta_run_exp"]
        .mean()
        .reset_index()
    )
    count_means["count"] = (
        count_means["balls"].astype(str) + "-" + count_means["strikes"].astype(str)
    )

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#E63946" if v > 0 else "#457B9D" for v in count_means["delta_run_exp"]]
    ax.bar(count_means["count"], count_means["delta_run_exp"], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Count (Balls-Strikes)", fontsize=11)
    ax.set_ylabel("Mean Delta Run Expectancy", fontsize=11)
    ax.set_title("Run Value by Count", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig
