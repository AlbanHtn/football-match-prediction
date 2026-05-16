"""Market-based prediction baselines for benchmarking models."""

import numpy as np
import pandas as pd


def market_baseline_accuracy(
    y_true: pd.Series | np.ndarray,
    odds: pd.DataFrame,
    outcome_to_odd: dict[str, str] | None = None,
) -> float:
    """Compute the accuracy of the naive 'bet on the favorite' strategy.

    The baseline always predicts the outcome with the lowest odds (= the
    bookmaker's favorite). This is the standard reference point used in the
    literature for match outcome prediction.

    Args:
        y_true: Ground-truth match outcomes.
        odds: DataFrame with odds columns, same index as ``y_true``.
        outcome_to_odd: Mapping from label to odds column name.

    Returns:
        Accuracy as a float in [0, 1].
    """
    if outcome_to_odd is None:
        outcome_to_odd = {"H": "OddHome", "D": "OddDraw", "A": "OddAway"}

    odd_cols = list(outcome_to_odd.values())
    labels = list(outcome_to_odd.keys())

    # Row-wise argmin of odds = favorite
    min_idx = odds[odd_cols].values.argmin(axis=1)
    y_pred = np.array([labels[i] for i in min_idx])

    return float((np.asarray(y_true) == y_pred).mean())
