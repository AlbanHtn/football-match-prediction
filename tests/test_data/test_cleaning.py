"""Tests for data cleaning utilities."""

import pandas as pd
import pytest

from football_prediction.data.cleaning import (
    BIG5_DIVISIONS,
    apply_completeness_filter,
    filter_big5,
    filter_post_year,
)


def _make_matches(divisions: list[str], years: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "Division": divisions,
        "MatchDate": pd.to_datetime([f"{y}-08-01" for y in years]),
        "HomeTeam": ["A"] * len(divisions),
        "AwayTeam": ["B"] * len(divisions),
        "FTResult": ["H"] * len(divisions),
    })


class TestFilterBig5:
    def test_keeps_only_big5(self):
        df = _make_matches(["E0", "F1", "B1", "SP1", "D2"], [2020] * 5)
        result = filter_big5(df)
        assert set(result["Division"].unique()) <= set(BIG5_DIVISIONS)
        assert len(result) == 3  # E0, F1, SP1

    def test_empty_input_returns_empty(self):
        df = _make_matches([], [])
        result = filter_big5(df)
        assert len(result) == 0

    def test_all_big5_keeps_all(self):
        df = _make_matches(list(BIG5_DIVISIONS), [2020] * len(BIG5_DIVISIONS))
        result = filter_big5(df)
        assert len(result) == len(BIG5_DIVISIONS)


class TestFilterPostYear:
    def test_excludes_earlier_years(self):
        df = _make_matches(["E0"] * 4, [2008, 2009, 2010, 2015])
        result = filter_post_year(df, min_year=2010)
        assert all(result["MatchDate"].dt.year >= 2010)
        assert len(result) == 2

    def test_inclusive_boundary(self):
        df = _make_matches(["E0"] * 2, [2010, 2011])
        result = filter_post_year(df, min_year=2010)
        assert len(result) == 2

    def test_no_rows_match(self):
        df = _make_matches(["E0"] * 3, [2000, 2001, 2002])
        result = filter_post_year(df, min_year=2020)
        assert len(result) == 0


class TestApplyCompletenessFilter:
    def test_drops_rows_with_nulls(self):
        df = pd.DataFrame({
            "a": [1.0, None, 3.0],
            "b": [4.0, 5.0, None],
            "c": [7.0, 8.0, 9.0],
        })
        result = apply_completeness_filter(df, columns=["a", "b"])
        assert len(result) == 1
        assert result.iloc[0]["a"] == 1.0

    def test_no_nulls_keeps_all(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = apply_completeness_filter(df, columns=["a", "b"])
        assert len(result) == 2

    def test_ignores_unchecked_columns(self):
        df = pd.DataFrame({"a": [1.0, None], "b": [None, 4.0]})
        result = apply_completeness_filter(df, columns=["a"])
        # Only row 0 has a non-null 'a'
        assert len(result) == 1
