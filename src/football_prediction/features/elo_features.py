"""Elo-based feature engineering."""

import pandas as pd


def add_elo_features(
    df: pd.DataFrame,
    home_elo_col: str = "HomeEloSnap",
    away_elo_col: str = "AwayEloSnap",
) -> pd.DataFrame:
    """Compute derived Elo features from raw Elo snapshots.

    Features added:
    - ``EloDiff``: HomeElo − AwayElo (positive = home stronger)
    - ``EloTotal``: sum of both Elo ratings
    - ``EloAdvantage``: EloDiff / EloTotal (normalized, scale-independent)

    Args:
        df: DataFrame with Elo snapshot columns.
        home_elo_col: Column containing the home team's Elo rating.
        away_elo_col: Column containing the away team's Elo rating.

    Returns:
        DataFrame with three new columns added.
    """
    df = df.copy()
    df["EloDiff"] = df[home_elo_col] - df[away_elo_col]
    df["EloTotal"] = df[home_elo_col] + df[away_elo_col]
    # Guard against division by zero for rows where both Elo values are 0
    df["EloAdvantage"] = df["EloDiff"] / df["EloTotal"].replace(0, float("nan"))
    return df


def add_elo_change_features(
    df: pd.DataFrame,
    elo: pd.DataFrame,
    date_col: str = "MatchDate",
    home_col: str = "HomeTeam",
    away_col: str = "AwayTeam",
    lag_months: tuple[int, ...] = (1, 2),
) -> pd.DataFrame:
    """Compute Elo momentum by comparing current Elo to N months ago.

    For each lag in ``lag_months``, computes:
    - ``EloChange{lag}Home``: current HomeElo minus HomeElo N months prior
    - ``EloChange{lag}Away``: current AwayElo minus AwayElo N months prior

    Args:
        df: Matches DataFrame with Elo snapshots already attached.
        elo: Raw Elo ratings DataFrame (Club, Date, Elo).
        date_col: Match date column.
        home_col: Home team name column.
        away_col: Away team name column.
        lag_months: Number of months to look back for each feature.

    Returns:
        DataFrame with Elo change columns added.
    """
    df = df.copy()
    elo_sorted = elo.sort_values("Date")

    for lag in lag_months:
        lag_offset = pd.DateOffset(months=lag)
        lagged_date_col = f"_lag_date_{lag}"
        df[lagged_date_col] = df[date_col] - lag_offset

        elo_lag = elo_sorted.rename(columns={"Date": lagged_date_col, "Elo": f"_elo_lag_{lag}"})

        for team_col, suffix in [(home_col, "Home"), (away_col, "Away")]:
            snap_col = f"HomeEloSnap" if suffix == "Home" else "AwayEloSnap"
            current_col = f"EloChange{lag}{suffix}"

            lagged = elo_lag.rename(columns={"Club": team_col})[[team_col, lagged_date_col, f"_elo_lag_{lag}"]]
            merged = pd.merge_asof(
                df[[team_col, lagged_date_col]].sort_values(lagged_date_col),
                lagged.sort_values(lagged_date_col),
                on=lagged_date_col,
                by=team_col,
                direction="backward",
            )
            merged = merged.set_index(df.sort_values(date_col).index)
            df[current_col] = df[snap_col] - merged[f"_elo_lag_{lag}"]

        df = df.drop(columns=[lagged_date_col])

    return df
