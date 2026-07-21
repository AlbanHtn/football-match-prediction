"""End-to-end training pipeline.

Usage:
    python -m football_prediction.pipeline [--model {rf,lr,xgb}] [--big5-only]
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from .config.settings import DataPaths, ModelConfig
from .data.cleaning import apply_completeness_filter, filter_big5, filter_post_year
from .data.ingestion import load_elo, load_matches
from .data.merging import merge_matches_with_elo
from .evaluation.metrics import compute_classification_metrics, compute_roi
from .features.elo_features import add_elo_features
from .features.form_features import add_form_momentum, add_form_rolling_stats
from .features.odds_features import add_implied_probabilities
from .models.base import BaseMatchPredictor
from .models.logistic_regression import LogisticRegressionPredictor
from .models.random_forest import RandomForestPredictor
from .models.xgboost_model import XGBoostPredictor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

_MODELS = {
    "rf": RandomForestPredictor,
    "lr": LogisticRegressionPredictor,
    "xgb": XGBoostPredictor,
}

_FEATURE_COLS = [
    "EloDiff", "EloTotal", "EloAdvantage",
    "Form3Home", "Form3Away", "Form5Home", "Form5Away",
    "FormMomentumHome", "FormMomentumAway",
    "ImpliedProbHome", "ImpliedProbDraw", "ImpliedProbAway",
]

_ODDS_COLS = ["OddHome", "OddDraw", "OddAway"]
_ROLLING_STAT_COLS = ["HomeShots", "HomeCorners", "HomeYellow"]


def _build_model(name: str, feature_columns: list[str]) -> BaseMatchPredictor:
    if name == "rf":
        return RandomForestPredictor(feature_columns=feature_columns)
    if name == "lr":
        return LogisticRegressionPredictor(feature_columns=feature_columns)
    if name == "xgb":
        return XGBoostPredictor(feature_columns=feature_columns)
    raise ValueError(f"Unknown model: {name}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Football match prediction pipeline")
    p.add_argument("--model", choices=list(_MODELS), default="xgb")
    p.add_argument("--big5-only", action="store_true", default=True)
    p.add_argument("--min-year", type=int, default=ModelConfig().min_season_year)
    p.add_argument("--cv-folds", type=int, default=ModelConfig().n_cv_splits)
    return p.parse_args(argv)


def build_features(df: pd.DataFrame, odds_cols: list[str] = _ODDS_COLS) -> pd.DataFrame:
    df = add_elo_features(df)
    df = add_form_rolling_stats(df, stat_columns=_ROLLING_STAT_COLS)
    df = add_form_momentum(df)
    available_odds = [c for c in odds_cols if c in df.columns]
    if len(available_odds) == 3:
        df = add_implied_probabilities(
            df, odd_home=odds_cols[0], odd_draw=odds_cols[1], odd_away=odds_cols[2]
        )
    return df


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    paths = DataPaths()

    logger.info("Loading data …")
    # Prefer raw/ if present, otherwise fall back to legacy Donnees/
    matches_path = paths.raw_matches if paths.raw_matches.exists() else paths.legacy_matches
    elo_path = paths.raw_elo if paths.raw_elo.exists() else paths.legacy_elo
    matches = load_matches(matches_path)
    elo = load_elo(elo_path)

    if args.big5_only:
        matches = filter_big5(matches)
    matches = filter_post_year(matches, min_year=args.min_year)
    matches = merge_matches_with_elo(matches, elo)

    logger.info("Engineering features …")
    df = build_features(matches)

    feature_cols = [c for c in _FEATURE_COLS if c in df.columns]
    df = apply_completeness_filter(df, columns=feature_cols + ["FTResult"])
    df = df.sort_values("MatchDate").reset_index(drop=True)

    logger.info("Using %d features on %d samples", len(feature_cols), len(df))

    # Temporal split: last 20% as hold-out test set
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    model = _build_model(args.model, feature_cols)
    model.fit(train)

    preds = model.predict(test)
    metrics = compute_classification_metrics(test["FTResult"].values, preds)

    logger.info("=== Results (%s, n_test=%d) ===", args.model.upper(), len(test))
    for k, v in metrics.items():
        logger.info("  %-20s %.4f", k, v)

    odds_available = all(c in test.columns for c in _ODDS_COLS)
    if odds_available:
        roi = compute_roi(test["FTResult"].values, preds, test[_ODDS_COLS])
        logger.info("  %-20s %.4f", "ROI", roi)


if __name__ == "__main__":
    run(sys.argv[1:])
