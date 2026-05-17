"""Unit tests for evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from football_prediction.evaluation.metrics import compute_classification_metrics, compute_roi
from football_prediction.evaluation.baseline import market_baseline_accuracy


def test_perfect_classifier():
    y = ["H", "D", "A", "H"]
    metrics = compute_classification_metrics(y, y)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)


def test_always_wrong():
    y_true = ["H", "H", "H"]
    y_pred = ["A", "A", "A"]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(0.0)


def test_market_baseline_picks_favorite():
    y_true = pd.Series(["H", "A"])
    odds = pd.DataFrame({
        "OddHome": [1.5, 3.0],   # row 0: home is fav; row 1: away is fav
        "OddDraw": [3.5, 3.5],
        "OddAway": [5.0, 2.0],
    })
    acc = market_baseline_accuracy(y_true, odds)
    assert acc == pytest.approx(1.0)


def test_roi_all_correct():
    y_true = np.array(["H", "H"])
    y_pred = np.array(["H", "H"])
    odds = pd.DataFrame({"OddHome": [2.0, 3.0], "OddDraw": [3.0, 3.0], "OddAway": [4.0, 4.0]})
    roi = compute_roi(y_true, y_pred, odds)
    # Staked 2, returned 2+3=5 → ROI = (5-2)/2 = 1.5
    assert roi == pytest.approx(1.5)


def test_roi_all_wrong():
    y_true = np.array(["A", "A"])
    y_pred = np.array(["H", "H"])
    odds = pd.DataFrame({"OddHome": [2.0, 3.0], "OddDraw": [3.0, 3.0], "OddAway": [4.0, 4.0]})
    roi = compute_roi(y_true, y_pred, odds)
    assert roi == pytest.approx(-1.0)
