"""Odds-derived feature engineering."""

import pandas as pd


def add_implied_probabilities(
    df: pd.DataFrame,
    odd_home: str = "OddHome",
    odd_draw: str = "OddDraw",
    odd_away: str = "OddAway",
) -> pd.DataFrame:
    """Convert bookmaker odds to implied probabilities and compute margin.

    Implied probability = 1 / odds. The sum exceeds 1 due to the bookmaker
    margin (overround), which itself is informative about market confidence.

    Features added:
    - ``ImpliedProbHome``, ``ImpliedProbDraw``, ``ImpliedProbAway``
    - ``ImpliedProbTotal``: sum of raw implied probs (> 1 by bookmaker margin)
    - ``BookmakerMargin``: ImpliedProbTotal − 1 (proxy for market uncertainty)
    - ``NormProbHome``, ``NormProbDraw``, ``NormProbAway``: margin-adjusted
      probabilities that sum to 1 (Shin-corrected approximation via division)

    Args:
        df: DataFrame with bookmaker odds columns.
        odd_home: Column name for home win odds.
        odd_draw: Column name for draw odds.
        odd_away: Column name for away win odds.

    Returns:
        DataFrame with implied probability columns added.
    """
    df = df.copy()

    df["ImpliedProbHome"] = 1.0 / df[odd_home]
    df["ImpliedProbDraw"] = 1.0 / df[odd_draw]
    df["ImpliedProbAway"] = 1.0 / df[odd_away]

    df["ImpliedProbTotal"] = (
        df["ImpliedProbHome"] + df["ImpliedProbDraw"] + df["ImpliedProbAway"]
    )
    df["BookmakerMargin"] = df["ImpliedProbTotal"] - 1.0

    # Normalize to sum to 1 (removes bookmaker margin)
    total = df["ImpliedProbTotal"]
    df["NormProbHome"] = df["ImpliedProbHome"] / total
    df["NormProbDraw"] = df["ImpliedProbDraw"] / total
    df["NormProbAway"] = df["ImpliedProbAway"] / total

    return df
