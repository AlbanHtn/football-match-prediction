"""Merge match data with Elo ratings using asof (backward) join."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def merge_matches_with_elo(
    matches: pd.DataFrame,
    elo: pd.DataFrame,
    home_col: str = "HomeTeam",
    away_col: str = "AwayTeam",
    date_col: str = "MatchDate",
) -> pd.DataFrame:
    """Attach the most recent Elo rating to each team before each match.

    Uses ``pd.merge_asof`` with ``direction='backward'`` so that only Elo
    snapshots taken *before* the match date are considered — preventing any
    look-ahead bias.

    Args:
        matches: Raw match DataFrame (output of ``load_matches``).
        elo: Raw Elo DataFrame (output of ``load_elo``).
        home_col: Column name for the home team.
        away_col: Column name for the away team.
        date_col: Column name for the match date.

    Returns:
        DataFrame with ``HomeEloSnap`` and ``AwayEloSnap`` columns added.
    """
    matches = matches.copy().sort_values(date_col).reset_index(drop=True)
    elo_sorted = elo.sort_values("Date").reset_index(drop=True)

    logger.info("Merging %d matches with Elo ratings …", len(matches))

    # Home Elo
    merged = _merge_team_elo(matches, elo_sorted, team_col=home_col, date_col=date_col, suffix="Home")
    # Away Elo
    merged = _merge_team_elo(merged, elo_sorted, team_col=away_col, date_col=date_col, suffix="Away")

    home_coverage = merged["HomeEloSnap"].notna().mean()
    away_coverage = merged["AwayEloSnap"].notna().mean()
    logger.info(
        "Elo coverage — Home: %.1f%%  Away: %.1f%%",
        home_coverage * 100,
        away_coverage * 100,
    )
    return merged


def _merge_team_elo(
    matches: pd.DataFrame,
    elo: pd.DataFrame,
    team_col: str,
    date_col: str,
    suffix: str,
) -> pd.DataFrame:
    """Attach the most recent Elo rating for a single team column."""
    elo_renamed = elo.rename(columns={"Club": team_col, "Date": date_col, "Elo": f"{suffix}EloSnap"})
    elo_renamed = elo_renamed[[team_col, date_col, f"{suffix}EloSnap"]]

    merged = pd.merge_asof(
        matches,
        elo_renamed,
        on=date_col,
        by=team_col,
        direction="backward",
    )
    return merged
