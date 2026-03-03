# Bayesian Hierarchical Model for Pitch-Level Expected Run Value

A Bayesian hierarchical regression that estimates the expected run value of every pitch thrown in the 2025 MLB season, using Statcast tracking data from Baseball Savant.

The model produces **posterior distributions**, not point estimates. Every prediction comes with calibrated uncertainty.

## What This Does

Takes 700K+ pitches of Statcast data (velocity, spin, movement, plate location, game state) and fits a hierarchical model in PyMC where:

- Each pitcher gets a **random intercept** drawn from a league-level distribution
- **Partial pooling** shrinks noisy estimates for low-sample pitchers toward the league average while letting high-volume pitchers express their true tendencies
- Fixed effects capture the impact of pitch physics, count leverage, platoon matchups, and location
- The target variable is `delta_run_exp`: the change in run expectancy from each pitch

## Why Hierarchical

MLB has ~800 pitchers in a season. Some throw 3,000+ pitches. Some throw 50. A model that treats each pitcher independently will overfit on the small samples and underfit on the large ones. Hierarchical modeling solves this by borrowing strength across the population. The math handles what your intuition already knows: a guy who threw 50 pitches and looked elite probably isn't actually that elite.

## Project Structure

```
statcast-bayesian-pitch-model/
├── notebooks/
│   ├── 01_data_acquisition.ipynb       # Pull + clean Statcast data
│   ├── 02_eda_and_feature_engineering.ipynb  # Visualization + features
│   ├── 03_bayesian_model.ipynb         # PyMC model build + sampling
│   └── 04_results_and_diagnostics.ipynb # Posteriors, baselines, Dodgers analysis
├── src/
│   ├── data.py          # pybaseball pull, cleaning, parquet caching
│   ├── features.py      # Platoon, count leverage, stuff composite, zones
│   ├── baseline.py      # ML baselines (Linear, RF, GBM) with cross-validation
│   ├── model.py         # PyMC hierarchical model definition
│   └── visualize.py     # Pitch heatmaps, forest plots, posterior checks
├── tests/
│   ├── test_data.py     # Data pipeline validation
│   ├── test_features.py # Feature engineering validation
│   ├── test_baseline.py # ML baseline output checks
│   └── test_model.py    # Model compilation and structure tests
├── data/                # Generated locally, not committed
├── figures/             # Output visualizations
├── pyproject.toml       # Dependencies + tool config
└── .github/workflows/
    └── ci.yml           # Ruff lint + pytest on push
```

## Features Engineered

| Feature | Description | Baseball Rationale |
|---------|-------------|-------------------|
| Platoon advantage | Same-hand matchup binary flag | RHP vs RHB is a different game than RHP vs LHB |
| Count leverage | Numeric encoding of ball-strike state | 0-2 and 3-0 are not the same pitch |
| Stuff composite | Z-scored velocity + movement within pitch type | A 95 mph fastball with 18" of rise is nasty |
| Location zones | Heart / edge / chase / waste classification | Middle-middle and low-and-away produce different outcomes |
| Base state | Integer 0-7 encoding of runner configuration | Runners on base change the run expectancy landscape |

## Model Specification

```
y ~ Normal(mu, sigma)

mu = alpha_pitcher[j] + X * beta

alpha_pitcher[j] ~ Normal(mu_alpha, sigma_alpha)   # pitcher random effects
mu_alpha ~ Normal(0, 0.1)                           # league-level mean
sigma_alpha ~ HalfNormal(0.05)                      # pitcher variance

beta ~ Normal(0, 0.1)                               # fixed effects
sigma ~ HalfNormal(0.1)                             # observation noise
```

Non-centered parameterization for clean MCMC geometry. Sampled with NUTS.

## Setup

```bash
# Clone
git clone https://github.com/angelrios97/statcast-bayesian-pitch-model.git
cd statcast-bayesian-pitch-model

# Install
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Start with notebook 01
jupyter notebook notebooks/01_data_acquisition.ipynb
```

## Requirements

- Python 3.10+
- PyMC 5.10+ (NUTS sampler, ArviZ integration)
- pybaseball (Statcast API wrapper)
- See `pyproject.toml` for full dependency list

## Diagnostics

The model includes full MCMC diagnostic checks at every step:

- **R-hat < 1.01** across all parameters (chain convergence)
- **ESS > 400** for reliable posterior estimates
- **Zero divergences** (sampler geometry)
- **Posterior predictive checks** (model reproduces observed data distribution)
- **Trace plots and rank plots** (visual convergence assessment)

## License

MIT
