# Football Match Prediction

Predicting football match outcomes (Home Win / Draw / Away Win) using historical match data, Elo ratings, betting odds, and rolling team form statistics.

**Dataset**: ~470,000 matches across 27 countries and 42 leagues (2000–2025)  
**Best model accuracy**: 57.15% (XGBoost with odds features) vs. 53.63% market baseline

---

## Architecture

```
football-prediction/
├── src/football_prediction/   # Core Python package
│   ├── config/                # Centralized path and model configuration
│   ├── data/                  # Ingestion, merging, cleaning
│   ├── features/              # Feature engineering (Elo, form, odds)
│   ├── models/                # RF, Logistic Regression, XGBoost
│   └── evaluation/            # Metrics, ROI simulation, market baseline
├── notebooks/                 # Analysis and experimentation notebooks
├── data/
│   ├── raw/                   # Source CSVs (gitignored — see below)
│   ├── interim/               # Intermediate outputs (gitignored)
│   └── processed/             # Analysis-ready datasets (gitignored)
├── references/                # Small reference tables (versioned)
├── models/                    # Serialized trained models (gitignored)
├── reports/eda/               # ydata-profiling HTML reports (gitignored)
└── tests/                     # Unit tests
```

## Data Sources

| File | Source | Size | Description |
|------|--------|------|-------------|
| `Matches.csv` | [Football-Data.co.uk](https://www.football-data.co.uk/) | ~37 MB | Match results and statistics (2000–2025) |
| `EloRatings.csv` | [ClubElo](https://www.clubelo.com/) | ~7.5 MB | Bi-monthly Elo snapshots for ~500 clubs |

Data files are **not versioned** (too large for git). Place them in `data/raw/` before running the pipeline.

See [`README_DB_Elo_Match.md`](README_DB_Elo_Match.md) for full column documentation.

## Installation

```bash
git clone https://github.com/AlbanHtn/Projet_5A_Prediction_matchs.git
cd Projet_5A_Prediction_matchs

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

## Data Setup

Place the raw source files:
```
data/raw/Matches.csv
data/raw/EloRatings.csv
```

Then run the preparation notebook:
```bash
jupyter notebook notebooks/01_data_preparation.ipynb
```

This produces the processed datasets in `data/interim/` and `data/processed/`.

## Pipeline Overview

```
data/raw/Matches.csv  ──┐
                         ├─→ [01_data_preparation] ─→ data/interim/Matches_with_Elo.csv
data/raw/EloRatings.csv ─┘                          ─→ data/processed/*.csv

data/processed/ ─→ [02_eda_univariate]       (distribution analysis)
               ─→ [03_eda_comparative]       (dataset version comparison)
               ─→ [04_modeling_comparison]   (RF / LR / XGBoost benchmarks)
```

## Models

All models target a 3-class prediction: `H` (Home Win), `D` (Draw), `A` (Away Win).

| Model | Accuracy (w/ odds) | Accuracy (blind) | Market baseline |
|-------|--------------------|------------------|-----------------|
| Logistic Regression | 55.68% | 52.45% | 53.63% |
| Random Forest | 54.82% | 52.47% | 53.63% |
| XGBoost | **57.15%** | 51.86% | 53.63% |

Training uses `TimeSeriesSplit` (5 folds) to respect temporal ordering.

## Running Tests

```bash
pytest
```

## Key Design Decisions

- **No data leakage**: post-match features (goals, shots, cards) are only used as lagged rolling stats from *previous* matches, never from the current game.
- **Temporal cross-validation**: `TimeSeriesSplit` instead of random k-fold — future matches cannot inform past predictions.
- **Market baseline as reference**: the naive "bet on the favorite" strategy is the true baseline, not random chance.
- **Elo merge strategy**: `merge_asof` with `direction='backward'` ensures only Elo snapshots *before* match date are used.

## Repository History

This project was originally developed as part of a 5th-year engineering school project (Guillet & Haton). The current structure reflects a professional refactoring of the original notebook-centric codebase.
