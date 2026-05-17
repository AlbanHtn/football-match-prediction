"""Unit tests for Elo feature engineering."""

import pandas as pd
import pytest

from football_prediction.features.elo_features import add_elo_features


@pytest.fixture
def sample_matches() -> pd.DataFrame:
    return pd.DataFrame({
        "HomeEloSnap": [1800.0, 1500.0, 1600.0],
        "AwayEloSnap": [1600.0, 1700.0, 1600.0],
    })


def test_elo_diff(sample_matches):
    result = add_elo_features(sample_matches)
    expected = [200.0, -200.0, 0.0]
    assert list(result["EloDiff"]) == expected


def test_elo_total(sample_matches):
    result = add_elo_features(sample_matches)
    expected = [3400.0, 3200.0, 3200.0]
    assert list(result["EloTotal"]) == expected


def test_elo_advantage_range(sample_matches):
    result = add_elo_features(sample_matches)
    assert result["EloAdvantage"].between(-1, 1).all()


def test_elo_advantage_equal_teams(sample_matches):
    result = add_elo_features(sample_matches)
    assert result["EloAdvantage"].iloc[2] == pytest.approx(0.0)


def test_does_not_modify_input(sample_matches):
    original_cols = list(sample_matches.columns)
    add_elo_features(sample_matches)
    assert list(sample_matches.columns) == original_cols
