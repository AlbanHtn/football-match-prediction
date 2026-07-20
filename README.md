# ⚽ Football Match Prediction — Market Efficiency Study

[![CI](https://github.com/AlbanHtn/Projet_5A_Prediction_matchs/actions/workflows/ci.yml/badge.svg)](https://github.com/AlbanHtn/Projet_5A_Prediction_matchs/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-49%20passing-brightgreen.svg)](#running-tests)

**Can pre-match sports data (Elo ratings, rolling form, match statistics) predict football outcomes as well as bookmaker odds do — without using the odds themselves?**

End-to-end pipeline on **52,580 Big-5-league matches (2006–2025)**: data reconciliation, temporal feature engineering, and a strict anti-leakage evaluation protocol (`TimeSeriesSplit`, 5 folds) comparing Logistic Regression, Random Forest, and XGBoost against the market baseline.

> **Headline result:** the blind XGBoost model (no odds) reaches **51.9% accuracy**, capturing **96.7% of the market's predictive signal** (53.63% baseline) with strong calibration (**ECE = 1.7%**). A value-betting simulation on its predictions yields **ROI = −5.8%** — matching the bookmaker's margin almost exactly. This is not a failed strategy; it is empirical evidence for the **Efficient Market Hypothesis** in football betting markets, obtained honestly rather than through overfitting or leakage.

---

## Key Results

| Model | Accuracy — with odds | Accuracy — blind (no odds) |
|-------|----------------------|------------------------------|
| Market baseline *(bet the favorite)* | — | **53.63%** |
| Logistic Regression | 53.39% | 52.45% |
| Random Forest | 53.41% | 52.47% |
| **XGBoost** | 52.54% | **51.86%** |

*Averaged over 5 chronological `TimeSeriesSplit` folds spanning 2013–2025. "Blind" models use only sport-derived features (Elo, form, match stats) — no bookmaker odds.*

**Diagnostics — Blind XGBoost:**

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Accuracy | 51.86% | 96.7% of market baseline's accuracy (51.86 / 53.63) |
| Log Loss | 0.993 | — |
| Brier Score (Home) | 0.218 | — |
| Expected Calibration Error (ECE) | 1.7% | Predicted probabilities closely match observed outcome frequencies |
| ROI at 0% value threshold | **−5.79%** | Aligns with the ~5–6% bookmaker overround → no exploitable edge |

**Why a negative ROI is the right result, not a failure:** a model that "beats the market" on a public, decade-spanning dataset without foresight bias would be extraordinary — and worth treating with suspicion. Odds set by professional bookmakers already price in almost everything a public dataset can offer. Reproducing 96.7% of that signal from sport-only features (no market information) is a strong result; losing money at a rate that mirrors the bookmaker's own margin, instead of some unexplained anomaly, is the expected, honest outcome of a market with no obvious inefficiency. It is presented here as validation of methodology, not as a trading strategy.

---

## Business Context

Betting markets aggregate enormous information flows — team news, public sentiment, sharp bettor positioning — into a single number: the odds. Testing whether a data-driven model can approach or beat that number is a direct test of market efficiency, and a standard sanity check any quantitative sports-betting team needs its models to pass *before* trusting them with capital.

**Why this is a hard, honest test:**
- Football is low-scoring and high-variance — a single set-piece changes the outcome
- Draws are structurally difficult to predict (~26% of matches, weak correlation with pre-match signal — see per-class recall below)
- Bookmaker odds already encode nearly all public information; beating them consistently would imply either overfitting, leakage, or a genuine (rare) edge
- Any rolling-form or Elo feature computed carelessly introduces look-ahead bias and produces artificially inflated accuracy — the opposite of what happened here

---

## Methodology

### Anti-Leakage Protocol — Three Independent Mechanisms

1. **Elo ratings** — merged via `pd.merge_asof(direction='backward')`: only Elo snapshots dated *before* each match are eligible.
2. **Rolling form / match statistics** — all rolling windows (`k=5`) use `.shift(1)`: the current match's own result never leaks into its own features.
3. **Cross-validation** — `TimeSeriesSplit(n_splits=5)`: training folds always precede their validation fold chronologically. No shuffling, no random k-fold.

### Two Model Variants per Algorithm

- **V1 — with odds**: includes bookmaker implied probabilities as features. Used as an upper-bound benchmark — closely tracks the market by construction.
- **V2 — blind**: sport-only features (Elo, form, match statistics). This is the model whose results are reported above — it answers the real question: *does public sports data alone carry predictive signal?*

### Data Reconciliation

Club names across the match results source and the Elo ratings source do not match exactly (accents, abbreviations, translations). A **fuzzy-matching** pass (`rapidfuzz`) against a curated mapping table (`club_mapping_clean.csv`) raised Elo-feature completeness from **59.5% to 90.6%** on the Big-5 subset.

---

## Architecture

```
src/football_prediction/          # Installable Python package (src layout, PEP 517)
│
├── config/
│   └── settings.py               # DataPaths + ModelConfig dataclasses — no hardcoded paths
│
├── data/
│   ├── ingestion.py               # CSV loading with schema validation & typed parsing
│   ├── cleaning.py                # Big5 filter, year filter, completeness filter
│   └── merging.py                 # Elo join via merge_asof(direction='backward')
│
├── features/
│   ├── elo_features.py            # EloDiff, EloTotal, HomeEloAdvantage
│   ├── form_features.py           # Rolling win rates (3 & 5-match windows) + momentum
│   └── odds_features.py           # Implied probabilities, bookmaker margin
│
├── models/
│   ├── base.py                    # Abstract BaseMatchPredictor — fit / predict / save / load
│   ├── logistic_regression.py     # Multinomial LR in a StandardScaler Pipeline
│   ├── random_forest.py           # RandomForest (n=200, max_depth=8, min_leaf=20)
│   └── xgboost_model.py           # XGBoost with H/D/A ↔ 0/1/2 label mapping
│
├── evaluation/
│   ├── metrics.py                 # Accuracy, F1-macro, log loss, ROI simulation
│   └── baseline.py                # Market baseline: argmax(implied probabilities)
│
└── pipeline.py                    # CLI entry point: load → clean → features → train → eval
```

---

## Data Pipeline

```
  Matches.csv                 EloRatings.csv
  228,377 rows                242,591 snapshots
  Football-Data.co.uk         ClubElo.com
        │                           │
        └─────────────┬─────────────┘
                       │
    Fuzzy-matched club reconciliation (rapidfuzz)
    merge_asof(direction='backward') — no look-ahead
    Elo feature completeness: 59.5% → 90.6%
                       │
        ┌──────────────▼──────────────┐
        │      Feature Engineering    │
        │  · EloDiff / EloTotal       │
        │  · Rolling form, k=5 (lag-1)│  ← lag-1: current match excluded
        │  · Implied probabilities     │  ← odds-based features (V1 only)
        └──────────────┬──────────────┘
                       │
        Filter: Big 5 leagues, 2006–2025
        → 52,580 matches
                       │
        TimeSeriesSplit (5 folds, chronological)
                       │
        ┌──────────────▼──────────────┐
        │   3-Class Classification    │
        │   H (Home) / D / A (Away)   │
        │  V1 (with odds) vs V2 (blind)│
        └──────────────────────────────┘
```

---

## Dataset

| Source | File | Size | Content |
|--------|------|------|---------|
| [Football-Data.co.uk](https://www.football-data.co.uk) | `Matches.csv` | 228,377 rows | Match results & statistics, multiple European leagues |
| [ClubElo.com](http://clubelo.com) | `EloRatings.csv` | 242,591 snapshots | Bi-monthly Elo ratings per club |

**Modeling dataset** (after Big-5 filter, post-2006, quality filtering): **52,580 matches** — Premier League · Ligue 1 · Bundesliga · Serie A · La Liga.

Data files are not versioned (too large for git). See [Installation](#installation) for setup.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Core ML** | scikit-learn, XGBoost, SHAP |
| **Data** | pandas, numpy, scipy, rapidfuzz |
| **Packaging** | pyproject.toml (PEP 517), src layout, setuptools |
| **Quality** | pytest (49 tests), ruff, mypy, GitHub Actions CI |
| **Profiling** | ydata-profiling |
| **Serialization** | joblib |

---

## Repository Structure

```
.
├── src/football_prediction/    # Core Python package (see Architecture)
├── tests/                      # 49 unit tests
│   ├── test_data/               # Ingestion, cleaning, merging
│   ├── test_features/           # Elo features, odds features
│   └── test_models/             # Model roundtrip + evaluation metrics
├── Programmes/                  # Jupyter notebooks (data prep, EDA, modeling)
│   ├── P5A_Prepa_donnees_V2.ipynb      # ETL pipeline, fuzzy-match reconciliation, Elo merge
│   ├── P5A_EDA_V2.ipynb                # Univariate EDA
│   ├── P5A_EDA_comparative.ipynb       # Dataset version comparison
│   ├── P5A_EDA_YDataprofiling.ipynb    # Automated HTML profiling
│   ├── Modélisation_RF_V2.ipynb        # Random Forest — V1/V2 + SHAP
│   ├── Modélisation_RL_V2.ipynb        # Logistic Regression — V1/V2 + SHAP
│   └── Modélisation_XGB.ipynb          # XGBoost — V1/V2, calibration, ROI simulation
├── Donnees/                     # Reference tables (club mappings, division codes)
├── reports/                     # Academic project report (PDF)
├── .github/workflows/ci.yml    # CI: pytest + ruff + mypy on Python 3.11 & 3.12
├── pyproject.toml               # Project metadata + dependencies
└── requirements.txt              # Runtime dependencies
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
python -m football_prediction.pipeline --model xgb --min-year 2006
```

### Python API

```python
from pathlib import Path
from football_prediction.data import load_matches, load_elo, merge_matches_with_elo
from football_prediction.features import add_elo_features, add_form_rolling_stats
from football_prediction.models import XGBoostPredictor

matches = load_matches(Path("Donnees/Matches.csv"))
elo = load_elo(Path("Donnees/EloRatings.csv"))
df = merge_matches_with_elo(matches, elo)

df = add_elo_features(df)
df = add_form_rolling_stats(df)

FEATURES = ["EloDiff", "EloTotal", "HomeForm3", "AwayForm3", "HomeMomentum", "AwayMomentum"]
model = XGBoostPredictor(feature_columns=FEATURES)
model.fit(df)
predictions = model.predict(df)  # "H" / "D" / "A"

model.save(Path("models/xgb_blind_v1.joblib"))
```

### Running Tests

```bash
pytest                      # All 49 tests
pytest tests/test_data/     # Data pipeline tests only
```

---

## Skills Demonstrated

| Domain | Skills |
|--------|--------|
| **Machine Learning** | Multi-class classification, temporal CV (`TimeSeriesSplit`), ensemble methods (XGBoost, RF), probability calibration (ECE), SHAP explainability |
| **Quantitative Finance / Betting Analytics** | Implied-probability extraction, bookmaker margin analysis, value-betting simulation, market efficiency testing |
| **Data Engineering** | ETL pipeline design, temporal asof-joins, fuzzy string matching (`rapidfuzz`), multi-version dataset management |
| **Software Engineering** | Python packaging (src layout, PEP 517), abstract base classes, type annotations, CLI design, model serialization |
| **MLOps** | CI/CD (GitHub Actions), 49 unit tests across 3 layers, linting (ruff), static type checking (mypy) |
| **Statistical Rigor** | Anti-leakage design (3 independent mechanisms), appropriate baseline selection, honest reporting of a negative-edge result |

---

## Author

**Alban Haton** — Data/ML Engineer specializing in Sports Analytics
[GitHub](https://github.com/AlbanHtn) · [LinkedIn](https://www.linkedin.com/in/alban-haton) · alban.haton@gmail.com

*Developed during an Assistant Football Data Project Manager role at LFP Media (Ligue de Football Professionnel), building on an earlier engineering-school project.*
