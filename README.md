# ⚽ Football Match Prediction

[![CI](https://github.com/AlbanHtn/Projet_5A_Prediction_matchs/actions/workflows/ci.yml/badge.svg)](https://github.com/AlbanHtn/Projet_5A_Prediction_matchs/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-49%20passing-brightgreen.svg)](#run-tests)

**3-class match outcome prediction — Home Win · Draw · Away Win — on 25 years of European football data.**

XGBoost achieves **57.2% accuracy**, outperforming the market baseline (53.6%) by +3.5 percentage points — a gap that translates into measurable positive ROI in betting simulations.

> Built as a production-grade Python package: installable, fully tested, type-annotated, and CI-verified on Python 3.11 & 3.12.

---

## Key Results

| Model | Accuracy (with odds) | Accuracy (no odds) | vs Market |
|-------|---------------------|--------------------|-----------|
| Logistic Regression | 55.68% | 52.45% | +2.1 pp |
| Random Forest | 54.82% | 52.47% | +1.2 pp |
| **XGBoost** | **57.15%** | 51.86% | **+3.5 pp** |
| Market baseline *(bet on the favorite)* | 53.63% | — | — |

> **Context:** Predicting 3-way football outcomes is intrinsically uncertain — academic literature consistently reports 55–60% as the ceiling for data-driven models. The relevant baseline is not random (33%) but the naive strategy of always backing the bookmaker's favorite (53.6%), which already encodes most publicly available information. Beating it by 3.5pp represents a genuine edge.

---

## Business Context

Bookmakers set odds that imply win probabilities. If a model consistently identifies when those implied probabilities are mis-calibrated, it generates statistical edge. This project tests whether publicly available pre-match information — team strength (Elo), recent form, and historical odds — contains exploitable signal *beyond* what the market already prices in.

**Why this problem is hard:**
- Football is a low-scoring, high-variance sport — a single set-piece changes the outcome
- Draw prediction is structurally difficult (~27% of matches, weak correlation with pre-match signals)
- Bookmakers aggregate enormous information flows; their implied probs are near-efficient
- Any rolling form or Elo feature computed naively will contain look-ahead bias and overfit

---

## Architecture

```
src/football_prediction/          # Installable Python package (src layout, PEP 517)
│
├── config/
│   └── settings.py               # DataPaths + ModelConfig dataclasses — no hardcoded paths
│
├── data/
│   ├── ingestion.py              # CSV loading with schema validation & typed parsing
│   ├── cleaning.py               # Big5 filter, year filter, completeness filter
│   └── merging.py                # Elo join via merge_asof(direction='backward')
│
├── features/
│   ├── elo_features.py           # EloDiff, EloTotal, HomeEloAdvantage
│   ├── form_features.py          # Rolling win rates (3 & 5-match windows) + momentum
│   └── odds_features.py          # Implied probabilities, bookmaker margin, normalized probs
│
├── models/
│   ├── base.py                   # Abstract BaseMatchPredictor — fit / predict / save / load
│   ├── logistic_regression.py    # Multinomial LR in a StandardScaler Pipeline
│   ├── random_forest.py          # RandomForest (n=200, max_depth=8, min_leaf=20)
│   └── xgboost_model.py          # XGBoost with H/D/A ↔ 0/1/2 label mapping
│
├── evaluation/
│   ├── metrics.py                # Accuracy, F1-macro, per-class F1, log loss, ROI sim
│   └── baseline.py               # Market baseline: argmax(implied probabilities)
│
└── pipeline.py                   # CLI entry point: load → clean → features → train → eval
```

---

## Data Pipeline

```
  Matches.csv                EloRatings.csv
  470k rows · 42 leagues     243k snapshots · ~500 clubs
  Football-Data.co.uk        ClubElo.com
        │                          │
        └────────────┬─────────────┘
                     │
           merge_asof(direction='backward')
           ← only pre-match Elo snapshots used
                     │
          ┌──────────▼──────────────────┐
          │      Feature Engineering    │
          │  · EloDiff / EloTotal       │
          │  · Form3 / Form5 (lag-1)    │  ← lag-1: current match excluded
          │  · Momentum (2×F3 − F5)     │
          │  · Implied probabilities     │
          │  · Bookmaker margin          │
          └──────────┬──────────────────┘
                     │
          TimeSeriesSplit (5 folds)
          ← chronological order enforced
                     │
          ┌──────────▼──────────────────┐
          │   3-Class Classification    │
          │   H (Home) / D / A (Away)   │
          └─────────────────────────────┘
```

---

## Feature Engineering

| Feature | Category | Description | Anti-leakage guarantee |
|---------|----------|-------------|----------------------|
| `EloDiff` | Elo | Home Elo − Away Elo at match date | `merge_asof` backward |
| `EloTotal` | Elo | Sum of both teams' Elo ratings | `merge_asof` backward |
| `HomeEloAdvantage` | Elo | HomeElo / (HomeElo + AwayElo) | `merge_asof` backward |
| `HomeForm3` / `AwayForm3` | Form | Win rate over last 3 matches | `.shift(1)` — lag-1 |
| `HomeForm5` / `AwayForm5` | Form | Win rate over last 5 matches | `.shift(1)` — lag-1 |
| `HomeMomentum` / `AwayMomentum` | Form | 2 × Form3 − Form5 (short vs long trend) | `.shift(1)` — lag-1 |
| `HomeImpliedProb` | Odds | Normalized implied probability (B365H) | Pre-match data |
| `DrawImpliedProb` | Odds | Normalized draw implied probability | Pre-match data |
| `AwayImpliedProb` | Odds | Normalized away implied probability | Pre-match data |
| `BookmakerMargin` | Odds | Σ raw implied probs − 1 (overround) | Pre-match data |

---

## Methodology

### No Data Leakage — Three Layers of Protection

Data leakage is the most common and most damaging mistake in sports prediction modelling. Three independent mechanisms prevent it here:

1. **Elo ratings** — merged with `pd.merge_asof(direction='backward')`: only Elo snapshots dated *before* each match are eligible.
2. **Rolling form** — all rolling statistics are computed with `.shift(1)`: the result of the current match is excluded from its own features.
3. **Cross-validation** — `TimeSeriesSplit(n_splits=5)`: training folds always precede their validation fold chronologically. No shuffling.

### Market Baseline

53.63% accuracy is achieved by a model that simply backs whichever team the bookmaker prices as favorite. This, not random (33%), is the correct baseline. Beating it requires capturing information the market has not already priced in.

### Explainability (SHAP)

All three models are analyzed with SHAP values in the notebooks — force plots for individual match explanations, summary plots for global feature importance. Bookmaker-implied probabilities dominate; Elo and form contribute independently.

---

## Dataset

| Source | File | Size | Content |
|--------|------|------|---------|
| [Football-Data.co.uk](https://www.football-data.co.uk) | `Matches.csv` | ~37 MB | 470k matches · 27 countries · 42 leagues · 2000–2025 |
| [ClubElo.com](http://clubelo.com) | `EloRatings.csv` | ~7.5 MB | Bi-monthly Elo snapshots · ~500 clubs |

**Processed dataset variants** (generated by the data preparation notebook):

| Dataset | Rows | Completeness | Scope |
|---------|------|-------------|-------|
| Full merged | 165,619 | 82% | All leagues, 2000–2025 |
| Recent | 138,312 | 92% | Post-2006, better Elo coverage |
| Big 5 | 52,580 | 97% | Premier League · Ligue 1 · Bundesliga · Serie A · La Liga |
| Clean | 8,478 | 98.5% | Post-2024, all features available |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Core ML** | scikit-learn, XGBoost, SHAP |
| **Data** | pandas 2.0, numpy, scipy, rapidfuzz |
| **Packaging** | pyproject.toml (PEP 517), src layout, setuptools |
| **Quality** | pytest · 49 tests, ruff, mypy, GitHub Actions CI |
| **Profiling** | ydata-profiling (automated HTML EDA reports) |
| **Serialization** | joblib (model persistence) |

---

## Repository Structure

```
.
├── src/football_prediction/    # Core Python package (see Architecture)
├── tests/                      # 49 unit tests
│   ├── test_data/              # Ingestion, cleaning, merging
│   ├── test_features/          # Elo features, odds features
│   └── test_models/            # Model roundtrip + evaluation metrics
├── Programmes/                 # Jupyter notebooks
│   ├── P5A_Prepa_donnees_V2.ipynb      # ETL pipeline + Elo merge
│   ├── P5A_EDA_V2.ipynb                # Univariate EDA
│   ├── P5A_EDA_comparative.ipynb       # Dataset version comparison
│   ├── P5A_EDA_YDataprofiling.ipynb    # Automated HTML profiling
│   ├── Modélisation_RF_V2.ipynb        # Random Forest + SHAP
│   ├── Modélisation_RL_V2.ipynb        # Logistic Regression + SHAP
│   └── Modélisation_XGB.ipynb          # XGBoost + benchmarking
├── Donnees/                    # Reference tables (club mappings, division codes)
├── reports/                    # Academic project report (PDF)
├── .github/workflows/ci.yml   # CI: pytest + ruff + mypy on Python 3.11 & 3.12
├── pyproject.toml              # Project metadata + dependencies
└── requirements.txt            # Runtime dependencies
```

---

## Installation

```bash
git clone https://github.com/AlbanHtn/Projet_5A_Prediction_matchs.git
cd Projet_5A_Prediction_matchs

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

**Data setup** — place raw files in `Donnees/`:
- `Matches.csv` from [football-data.co.uk/data.php](https://www.football-data.co.uk/data.php)
- `EloRatings.csv` from [clubelo.com/ELO](http://clubelo.com/ELO)

Then run the data preparation notebook: `Programmes/P5A_Prepa_donnees_V2.ipynb`

---

## Usage

### CLI Pipeline

```bash
# Default: XGBoost, Big 5 leagues, min year from config
python -m football_prediction.pipeline

# Logistic Regression on all leagues from 2012 onwards
python -m football_prediction.pipeline --model lr --min-year 2012

# Random Forest with 3 CV folds
python -m football_prediction.pipeline --model rf --cv-folds 3
```

### Python API

```python
from pathlib import Path
from football_prediction.data import load_matches, load_elo, merge_matches_with_elo
from football_prediction.features import add_elo_features, add_form_rolling_stats
from football_prediction.models import XGBoostPredictor

# Load & merge
matches = load_matches(Path("Donnees/Matches.csv"))
elo = load_elo(Path("Donnees/EloRatings.csv"))
df = merge_matches_with_elo(matches, elo)

# Feature engineering (no leakage)
df = add_elo_features(df)
df = add_form_rolling_stats(df)

# Train & predict
FEATURES = ["EloDiff", "EloTotal", "HomeForm3", "AwayForm3", "HomeMomentum", "AwayMomentum"]
model = XGBoostPredictor(feature_columns=FEATURES)
model.fit(df)
predictions = model.predict(df)  # array of "H", "D", "A"

# Persist
model.save(Path("models/xgb_v1.joblib"))
```

### Run Tests

```bash
pytest                      # All 49 tests
pytest tests/test_data/     # Data pipeline tests only
pytest -v --tb=short        # Verbose with short tracebacks
```

---

## Future Improvements

- [ ] Streamlit dashboard — live probability display for upcoming fixtures
- [ ] Head-to-head (H2H) features — historical win rate between the two specific teams
- [ ] Elo change velocity — short-term form signal from Elo momentum
- [ ] Calibration — Platt scaling or isotonic regression to align predicted probs with actual frequencies
- [ ] Hyperparameter tuning — nested CV with Optuna
- [ ] FastAPI endpoint — serve single-match predictions as a REST API

---

## Skills Demonstrated

| Domain | Skills |
|--------|--------|
| **Machine Learning** | Multi-class classification, temporal CV (`TimeSeriesSplit`), ensemble methods (XGBoost, RF), regularization, SHAP explainability |
| **Data Engineering** | ETL pipeline design, temporal asof-joins, fuzzy string matching (`rapidfuzz`), multi-version dataset management |
| **Software Engineering** | Python packaging (src layout, PEP 517), abstract base classes, type annotations, CLI design, model serialization |
| **MLOps** | CI/CD (GitHub Actions), 49 unit tests across 3 layers, linting (ruff), static type checking (mypy) |
| **Domain Knowledge** | Sports analytics, Elo rating systems, betting market mechanics, bookmaker margin analysis, ROI simulation |
| **Statistical Rigor** | Anti-leakage design (3 independent mechanisms), appropriate baseline selection, temporal data integrity |

---

## Author

**Alban Haton** · [GitHub @AlbanHtn](https://github.com/AlbanHtn)

*Final-year engineering school project — subsequently refactored to production-grade MLOps architecture.*  
*Original co-development with Simon Guillet.*
