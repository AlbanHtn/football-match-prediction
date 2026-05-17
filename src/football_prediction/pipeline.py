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
    "EloDiff", "EloTotal", "HomeEloAdvantage",
    "HomeForm3", "AwayForm3", "HomeForm5", "AwayForm5",
    "HomeMomentum", "AwayMomentum",
    "HomeImpliedProb", "DrawImpliedProb", "AwayImpliedProb",
]

_ODDS_COLS = ["B365H", "B365D", "B365A"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Football match prediction pipeline")
    p.add_argument("--model", choices=list(_MODELS), default="xgb")
    p.add_argument("--big5-only", action="store_true", default=True)
    p.add_argument("--min-year", type=int, default=ModelConfig.MIN_YEAR)
    p.add_argument("--cv-folds", type=int, default=ModelConfig.CV_FOLDS)
    return p.parse_args(argv)


def build_features(df: pd.DataFrame, odds_cols: list[str] = _ODDS_COLS) -> pd.DataFrame:
    df = add_elo_features(df)
    df = add_form_rolling_stats(df)
    df = add_form_momentum(df)
    available_odds = [c for c in odds_cols if c in df.columns]
    if len(available_odds) == 3:
        df = add_implied_probabilities(df, home_col=odds_cols[0], draw_col=odds_cols[1], away_col=odds_cols[2])
    return df


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    paths = DataPaths()

    logger.info("Loading data …")
    matches = load_matches(paths.matches_csv)
    elo = load_elo(paths.elo_csv)

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

    ModelClass = _MODELS[args.model]
    model = ModelClass(feature_columns=feature_cols)
    model.fit(train)

    preds = model.predict(test)
    metrics = compute_classification_metrics(test["FTResult"].values, preds)

    logger.info("=== Results (%s, n_test=%d) ===", args.model.upper(), len(test))
    for k, v in metrics.items():
        logger.info("  %-20s %.4f", k, v)

    odds_available = all(c in test.columns for c in _ODDS_COLS)
    if odds_available:
        roi = compute_roi(test["FTResult"].values, preds, test[_ODDS_COLS].values)
        logger.info("  %-20s %.4f", "ROI", roi)


if __name__ == "__main__":
    run(sys.argv[1:])
