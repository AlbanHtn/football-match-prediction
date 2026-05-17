"""Evaluation metrics for match outcome prediction."""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: tuple[str, ...] = ("H", "D", "A"),
) -> dict[str, float]:
    """Compute standard classification metrics for a 3-class prediction.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted class probabilities (n_samples × 3). Required for
            ``log_loss`` computation.
        labels: Ordered class labels matching the columns of ``y_proba``.

    Returns:
        Dictionary with ``accuracy``, ``f1_macro``, and optionally ``log_loss``.
    """
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_home": float(f1_score(y_true, y_pred, labels=["H"], average="micro", zero_division=0)),
        "f1_draw": float(f1_score(y_true, y_pred, labels=["D"], average="micro", zero_division=0)),
        "f1_away": float(f1_score(y_true, y_pred, labels=["A"], average="micro", zero_division=0)),
    }
    if y_proba is not None:
        metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=list(labels)))
    return metrics


def compute_roi(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    odds: pd.DataFrame,
    outcome_to_odd: dict[str, str] | None = None,
    threshold: float = 0.0,
    pred_proba: np.ndarray | None = None,
) -> float:
    """Simulate a flat-stake betting strategy and return ROI.

    Bets are placed on every predicted outcome. If ``threshold`` > 0 and
    ``pred_proba`` is provided, only bets where the model's confidence
    exceeds the threshold are placed.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        odds: DataFrame with odds columns (same index as y_true).
        outcome_to_odd: Mapping from label to odds column name.
            Defaults to ``{"H": "OddHome", "D": "OddDraw", "A": "OddAway"}``.
        threshold: Minimum predicted probability to place a bet (0 = always bet).
        pred_proba: Predicted probabilities (n_samples × 3). Required when
            ``threshold`` > 0.

    Returns:
        Return on investment as a fraction (e.g. -0.05 = −5% ROI).
    """
    if outcome_to_odd is None:
        outcome_to_odd = {"H": "OddHome", "D": "OddDraw", "A": "OddAway"}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    total_staked = 0.0
    total_returned = 0.0

    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if threshold > 0.0 and pred_proba is not None:
            # Only bet if model confidence exceeds threshold
            pred_classes = list(outcome_to_odd.keys())
            pred_idx = pred_classes.index(pred) if pred in pred_classes else -1
            if pred_idx < 0 or pred_proba[i, pred_idx] < threshold:
                continue

        odd_col = outcome_to_odd.get(pred)
        if odd_col is None or odd_col not in odds.columns:
            continue

        odd = odds.iloc[i][odd_col]
        if pd.isna(odd) or odd <= 0:
            continue

        total_staked += 1.0
        if true == pred:
            total_returned += odd

    if total_staked == 0:
        return float("nan")

    return (total_returned - total_staked) / total_staked
