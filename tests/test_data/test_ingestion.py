"""Tests for data ingestion: schema validation and FileNotFoundError."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from football_prediction.data.ingestion import load_elo, load_matches


def _write_csv(df: pd.DataFrame) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    df.to_csv(tmp.name, index=False)
    return Path(tmp.name)


class TestLoadMatches:
    def test_loads_valid_file(self):
        df = pd.DataFrame({
            "Division": ["E0"],
            "MatchDate": ["2020-08-01"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTResult": ["H"],
        })
        path = _write_csv(df)
        result = load_matches(path)
        assert "MatchDate" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["MatchDate"])

    def test_raises_if_file_missing(self):
        with pytest.raises(FileNotFoundError):
            load_matches(Path("/nonexistent/Matches.csv"))

    def test_raises_if_columns_missing(self):
        df = pd.DataFrame({"Division": ["E0"], "HomeTeam": ["Arsenal"]})
        path = _write_csv(df)
        with pytest.raises(ValueError, match="missing required columns"):
            load_matches(path)


class TestLoadElo:
    def test_loads_valid_file(self):
        df = pd.DataFrame({
            "Date": ["2020-01-01"],
            "Club": ["Arsenal"],
            "Elo": [1800.0],
        })
        path = _write_csv(df)
        result = load_elo(path)
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])
        assert result["Elo"].iloc[0] == 1800.0

    def test_raises_if_file_missing(self):
        with pytest.raises(FileNotFoundError):
            load_elo(Path("/nonexistent/EloRatings.csv"))

    def test_raises_if_columns_missing(self):
        df = pd.DataFrame({"Date": ["2020-01-01"], "Club": ["Arsenal"]})
        path = _write_csv(df)
        with pytest.raises(ValueError, match="missing required columns"):
            load_elo(path)
