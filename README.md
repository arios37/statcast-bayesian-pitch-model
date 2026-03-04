# Bayesian Hierarchical Model for Pitch-Level Expected Run Value

[![CI](https://github.com/arios37/statcast-bayesian-pitch-model/actions/workflows/ci.yml/badge.svg)](https://github.com/arios37/statcast-bayesian-pitch-model/actions/workflows/ci.yml)
[![Deploy](https://github.com/arios37/statcast-bayesian-pitch-model/actions/workflows/deploy.yml/badge.svg)](https://github.com/arios37/statcast-bayesian-pitch-model/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

A Bayesian hierarchical regression that estimates the expected run value of every pitch thrown in the 2025 MLB season, using Statcast tracking data from Baseball Savant. The model produces **posterior distributions**, not point estimates -- every prediction comes with calibrated uncertainty. Covers all 30 MLB teams and 864 pitchers.

**[Live App](https://arios37.github.io/statcast-bayesian-pitch-model/)** -- search any pitcher, explore their arsenal, posterior distribution, team shrinkage, and count leverage.

---

## Interactive Explorer

The React frontend lets you search any of the 864 MLB pitchers and explore four views:

| Tab | What It Shows |
|-----|--------------|
| **Arsenal** | Pitch movement scatter (horizontal vs. vertical break) with velocity, spin, usage, and DRE per pitch type |
| **Posterior** | Full posterior distribution of the pitcher's random intercept (alpha) with credible intervals and reference lines |
| **Shrinkage** | Team-level Bayesian shrinkage plot showing how partial pooling pulls raw means toward the league hyperprior |
| **Count Leverage** | Expected delta run value by ball-strike count using the pitcher's best pitch in each state |

Stat chips show team, ERA, IP, pitch count, posterior mean, 90% credible interval, and whether the estimate comes from MCMC or conjugate update.

---

## Results

### Posterior Predictive Check

The model's posterior predictive distribution (blue) envelopes the observed `delta_run_exp` distribution (black), confirming the model captures the data-generating process.

![Posterior Predictive Check](figures/ppc.png)

### Shrinkage Plot

Partial pooling in action. High-volume pitchers stay near the diagonal (little shrinkage). Low-volume pitchers are pulled toward the league mean (strong shrinkage). This is the core value proposition of hierarchical modeling over raw averages.

![Shrinkage Plot](figures/shrinkage_plot.png)

### Fixed Effects (Posterior Forest)

Standardized coefficients with 94% HDI. Coefficients with intervals entirely away from zero represent clear effects. Magnitudes are directly comparable because all predictors are standardized.

![Fixed Effects Forest](figures/fixed_effects_forest.png)

### Pitcher Random Effects (Top 40)

Each pitcher's posterior intercept distribution. Tight intervals = high-volume, reliable estimate. Wide intervals = low-volume, heavily regularized by the hierarchical prior.

![Pitcher Effects Forest](figures/pitcher_effects_forest.png)

---

## What This Does

A four-stage pipeline from raw Statcast data to Bayesian inference, plus a React frontend for exploration:

| Stage | What Happens | Key Output |
|-------|-------------|------------|
| **Extract** | Pull 700K+ pitches from Baseball Savant via `pybaseball` | Raw Statcast DataFrame |
| **Engineer** | Build model features: platoon splits, count leverage, stuff composite, location zones, base state | Scaled feature matrix with pitcher indices |
| **Model** | Fit PyMC hierarchical regression with pitcher random intercepts, NUTS sampling | ArviZ InferenceData with posterior + posterior predictive |
| **Export** | Extract MCMC traces + compute conjugate posteriors for all 864 pitchers | `posteriors.json` consumed by the frontend |
| **Visualize** | React app with searchable pitcher explorer, four interactive chart tabs | Deployed to GitHub Pages |

---

## Features Engineered

| Feature | Description | Baseball Rationale |
|---------|-------------|-------------------|
| Platoon advantage | Binary flag: same-hand pitcher-batter matchup | RHP vs RHB is a fundamentally different at-bat than RHP vs LHB |
| Count leverage | Numeric encoding of the 12 ball-strike states | A 3-0 pitch and a 0-2 pitch have completely different run expectancy implications |
| Stuff composite | Z-scored velocity + horizontal/vertical movement within pitch type | A 95 mph fastball with 18" of induced vertical break is elite stuff regardless of era |
| Location zones | Heart / edge / chase / waste classification from plate coordinates | Middle-middle fastballs and low-and-away sliders produce categorically different outcomes |
| Base state | Integer 0-7 encoding of all 8 runner configurations | Runners on base shift the entire run expectancy landscape for every pitch |

---

## Model Specification

```
y ~ Normal(mu, sigma)

mu = alpha_pitcher[j] + X * beta

alpha_pitcher[j] ~ Normal(mu_alpha, sigma_alpha)   # pitcher random effects
mu_alpha ~ Normal(0, 0.1)                           # league-level mean
sigma_alpha ~ HalfNormal(0.05)                      # between-pitcher variance

beta ~ Normal(0, 0.1)                               # fixed effects (10 predictors)
sigma ~ HalfNormal(0.1)                             # observation noise
```

Non-centered parameterization (`alpha_pitcher = mu_alpha + sigma_alpha * offset`) for clean MCMC geometry. Sampled with NUTS, 4 chains, 1000 draws + 500 tune.

### Why Hierarchical

MLB has ~800 pitchers in a season. Some throw 3,000+ pitches. Some throw 50. A model that treats each pitcher independently will overfit on the small samples and miss the signal on the large ones. Hierarchical modeling solves this by borrowing strength across the population. The math handles what your intuition already knows: a guy who threw 50 pitches and looked elite probably isn't actually that elite. The model knows how much to trust each sample.

### Conjugate Updates for Non-Subsample Pitchers

The MCMC model runs on a subsample of pitchers. The remaining 800+ pitchers get their posteriors via conjugate normal-normal updating using the model's estimated hyperprior (mu_alpha, sigma_alpha). This is analytically equivalent to what MCMC would produce for this model structure and runs in milliseconds.

---

## ML Baseline Comparison

The Bayesian model is compared against three frequentist baselines. The point isn't that Bayes wins on RMSE -- it's that the Bayesian model provides **full posterior distributions** while baselines give you a single number.

| Model | RMSE | MAE | R-squared | CV RMSE (mean) | CV RMSE (std) |
|-------|------|-----|-----------|----------------|---------------|
| Linear Regression | 0.2223 | 0.1197 | 0.0004 | 0.2224 | 0.0015 |
| Random Forest | 0.2073 | 0.1015 | 0.1311 | 0.2203 | 0.0013 |
| Gradient Boosting | 0.2006 | 0.1012 | 0.1868 | 0.2229 | 0.0015 |

---

## Diagnostics

Full MCMC diagnostic suite at every stage:

| Check | Threshold | Purpose |
|-------|-----------|---------|
| R-hat | < 1.01 | Chain convergence -- all chains exploring the same posterior |
| ESS (bulk) | > 400 | Effective sample size -- enough independent draws for reliable estimates |
| Divergences | 0 | Sampler geometry -- no pathological regions in the posterior |
| Posterior predictive | Visual | Model reproduces the observed `delta_run_exp` distribution |
| Trace plots | Visual | Chains are mixing well, no stuck regions |
| Shrinkage plot | Visual | Partial pooling behaves as expected across sample sizes |

---

## Project Structure

```
statcast-bayesian-pitch-model/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Ruff lint + pytest on push (Python 3.10, 3.11)
│       └── deploy.yml              # Build React app + deploy to GitHub Pages
├── frontend/
│   ├── src/
│   │   └── App.jsx                 # React app: search, arsenal, posterior, shrinkage, count leverage
│   ├── public/
│   │   └── data/
│   │       └── posteriors.json     # 864 pitchers, all 30 teams, posterior + arsenal data
│   ├── package.json
│   └── vite.config.js
├── notebooks/
│   ├── 01_data_acquisition.ipynb         # Pull + clean 2025 Statcast data via pybaseball
│   ├── 02_eda_and_feature_engineering.ipynb  # EDA visualizations + feature engineering
│   ├── 03_bayesian_model.ipynb           # PyMC model build + MCMC sampling
│   └── 04_results_and_diagnostics.ipynb  # Posteriors, baselines, Dodgers deep dive
├── src/
│   ├── data.py         # pybaseball pull, cleaning, parquet caching
│   ├── features.py     # Platoon, count leverage, stuff composite, location zones
│   ├── baseline.py     # ML baselines (Linear, RF, GBM) with cross-validation
│   ├── model.py        # PyMC hierarchical model definition + diagnostics
│   ├── visualize.py    # Pitch heatmaps, forest plots, posterior checks
│   └── export_posteriors.py  # Extract MCMC traces + conjugate updates to JSON
├── tests/
│   ├── test_data.py        # Data pipeline validation
│   ├── test_features.py    # Feature engineering (42 assertions)
│   ├── test_baseline.py    # ML baseline output format checks
│   └── test_model.py       # Model compilation, structure, coords
├── figures/            # Generated visualizations (committed)
├── data/               # Parquet + NetCDF files (gitignored, ~200MB)
├── Makefile            # test, lint, clean, export, frontend-build, deploy
├── pyproject.toml      # Dependencies + ruff/pytest config
├── LICENSE             # MIT
└── README.md
```

---

## Quick Start

### Backend (Model + Data Pipeline)

```bash
git clone https://github.com/arios37/statcast-bayesian-pitch-model.git
cd statcast-bayesian-pitch-model

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"

pytest tests/ -v
ruff check src/ tests/

# Start with notebook 01 (data pull takes ~5 min)
jupyter notebook notebooks/01_data_acquisition.ipynb
```

### Frontend (React Explorer)

```bash
cd frontend
npm install
npm run dev    # http://localhost:5173/statcast-bayesian-pitch-model/
```

### Make Targets

```bash
make test            # pytest tests/ -v
make lint            # ruff check src/ tests/
make export          # generate posteriors.json from model output
make frontend-build  # vite build
make deploy          # export + build (full pipeline)
make clean           # remove data/, figures/, __pycache__
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **PyMC 5.10+** | Probabilistic programming, NUTS sampler, posterior predictive |
| **ArviZ** | MCMC diagnostics, InferenceData, trace/forest/rank plots |
| **pybaseball** | Statcast API wrapper for Baseball Savant data |
| **scikit-learn** | ML baselines (Linear, RF, Gradient Boosting), cross-validation |
| **React + Vite** | Interactive frontend with searchable pitcher explorer |
| **Recharts** | Charts: scatter, area, composed bar, bar |
| **pandas / NumPy** | Data manipulation, feature engineering |
| **matplotlib / seaborn** | Static visualizations, pitch heatmaps |
| **pytest** | 42-test suite covering data, features, model, baselines |
| **ruff** | Linting (pycodestyle, pyflakes, isort, bugbear, naming) |
| **GitHub Actions** | CI (lint + test) and deploy (build + GitHub Pages) |

---

## License

[MIT](LICENSE)
