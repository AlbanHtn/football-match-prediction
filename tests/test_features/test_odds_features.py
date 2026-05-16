"""Unit tests for odds-derived feature engineering."""

import pandas as pd
import pytest

from football_prediction.features.odds_features import add_implied_probabilities


@pytest.fixture
def sample_odds() -> pd.DataFrame:
    # Typical match: slight home favorite
    return pd.DataFrame({
        "OddHome": [2.0, 1.5],
        "OddDraw": [3.4, 4.0],
        "OddAway": [4.0, 7.0],
    })


def test_implied_probs_computed(sample_odds):
    result = add_implied_probabilities(sample_odds)
    assert "ImpliedProbHome" in result.columns
    assert "ImpliedProbDraw" in result.columns
    assert "ImpliedProbAway" in result.columns


def test_implied_prob_values(sample_odds):
    result = add_implied_probabilities(sample_odds)
    assert result["ImpliedProbHome"].iloc[0] == pytest.approx(0.5, rel=1e-3)
    assert result["ImpliedProbAway"].iloc[0] == pytest.approx(0.25, rel=1e-3)


def test_bookmaker_margin_positive(sample_odds):
    result = add_implied_probabilities(sample_odds)
    assert (result["BookmakerMargin"] > 0).all()


def test_normalized_probs_sum_to_one(sample_odds):
    result = add_implied_probabilities(sample_odds)
    norm_sum = result["NormProbHome"] + result["NormProbDraw"] + result["NormProbAway"]
    assert norm_sum.round(10).eq(1.0).all()


def test_does_not_modify_input(sample_odds):
    original_cols = list(sample_odds.columns)
    add_implied_probabilities(sample_odds)
    assert list(sample_odds.columns) == original_cols
