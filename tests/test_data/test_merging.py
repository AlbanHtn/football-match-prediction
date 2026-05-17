"""Tests for merge_matches_with_elo (asof backward join)."""

import pandas as pd
import pytest

from football_prediction.data.merging import merge_matches_with_elo


def _make_matches() -> pd.DataFrame:
    return pd.DataFrame({
        "Division": ["E0", "E0", "E0"],
        "MatchDate": pd.to_datetime(["2020-08-15", "2020-09-01", "2020-10-01"]),
        "HomeTeam": ["Arsenal", "Chelsea", "Arsenal"],
        "AwayTeam": ["Chelsea", "Arsenal", "Liverpool"],
        "FTResult": ["H", "A", "D"],
    })


def _make_elo() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.to_datetime([
            "2020-01-01", "2020-07-01",  # Arsenal snapshots
            "2020-01-01", "2020-08-01",  # Chelsea snapshots
            "2020-01-01",                # Liverpool snapshot
        ]),
        "Club": ["Arsenal", "Arsenal", "Chelsea", "Chelsea", "Liverpool"],
        "Elo": [1800.0, 1820.0, 1750.0, 1760.0, 1700.0],
    })


class TestMergeMatchesWithElo:
    def test_output_has_elo_columns(self):
        matches = _make_matches()
        elo = _make_elo()
        result = merge_matches_with_elo(matches, elo)
        assert "HomeEloSnap" in result.columns
        assert "AwayEloSnap" in result.columns

    def test_backward_join_uses_latest_before_match(self):
        matches = _make_matches()
        elo = _make_elo()
        result = merge_matches_with_elo(matches, elo).sort_values("MatchDate").reset_index(drop=True)
        # Arsenal match on 2020-08-15 → last snapshot 2020-07-01 → Elo 1820
        arsenal_row = result[result["HomeTeam"] == "Arsenal"].iloc[0]
        assert arsenal_row["HomeEloSnap"] == pytest.approx(1820.0)

    def test_no_future_elo_leak(self):
        # Match on 2020-02-01, only Elo snapshot is 2020-07-01 (future) → should be NaT / NaN
        matches = pd.DataFrame({
            "Division": ["E0"],
            "MatchDate": pd.to_datetime(["2020-02-01"]),
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTResult": ["H"],
        })
        elo = pd.DataFrame({
            "Date": pd.to_datetime(["2020-07-01"]),
            "Club": ["Arsenal"],
            "Elo": [1820.0],
        })
        result = merge_matches_with_elo(matches, elo)
        assert pd.isna(result["HomeEloSnap"].iloc[0])

    def test_row_count_preserved(self):
        matches = _make_matches()
        elo = _make_elo()
        result = merge_matches_with_elo(matches, elo)
        assert len(result) == len(matches)
