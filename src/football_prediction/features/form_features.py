"""Rolling form and momentum feature engineering.

These features are computed using only *past* matches — no look-ahead.
All rolling windows are applied per-team and shifted by 1 to exclude the
current match from its own features.
"""

import pandas as pd


def add_form_momentum(
    df: pd.DataFrame,
    form3_home: str = "Form3Home",
    form3_away: str = "Form3Away",
    form5_home: str = "Form5Home",
    form5_away: str = "Form5Away",
) -> pd.DataFrame:
    """Compute form difference and momentum features.

    Momentum = 2 * Form3 - Form5  (recent 3 vs older 2 matches from Form5).
    A positive momentum means form is improving; negative means declining.

    Args:
        df: DataFrame with Form3/Form5 columns for both teams.

    Returns:
        DataFrame with added form-derived columns.
    """
    df = df.copy()
    df["Form3Diff"] = df[form3_home] - df[form3_away]
    df["Form5Diff"] = df[form5_home] - df[form5_away]
    # Momentum: Form3 − (Form5 − Form3)  →  2*Form3 − Form5
    df["FormMomentumHome"] = 2 * df[form3_home] - df[form5_home]
    df["FormMomentumAway"] = 2 * df[form3_away] - df[form5_away]
    return df


def add_form_rolling_stats(
    df: pd.DataFrame,
    stat_columns: list[str],
    team_cols: tuple[str, str] = ("HomeTeam", "AwayTeam"),
    date_col: str = "MatchDate",
    windows: tuple[int, ...] = (5,),
) -> pd.DataFrame:
    """Compute rolling means of game statistics per team, lag-1 (no leakage).

    For each statistic in ``stat_columns`` and each ``window``, computes the
    rolling mean over the last ``window`` matches for each team, using only
    data *before* the current match.

    This is used in the modeling notebooks to add technical rolling stats
    (shots, corners, cards) as pre-match features — NOTE: these stats are
    post-match values from *previous* games, not from the current game.

    Args:
        df: DataFrame sorted by date with per-match statistics.
        stat_columns: Columns to compute rolling stats for (e.g. ``HomeShots``).
        team_cols: Tuple of (home_team_col, away_team_col).
        date_col: Date column used for sorting.
        windows: Rolling window sizes.

    Returns:
        DataFrame with rolling stat columns added (suffix: ``_RollN``).
    """
    df = df.copy().sort_values(date_col).reset_index(drop=True)

    for col in stat_columns:
        for window in windows:
            new_col = f"{col}_Roll{window}"
            # Compute per-team rolling mean, shifted by 1 to avoid leakage
            rolling_series = (
                df.groupby(df[[team_cols[0], team_cols[1]]].stack().name if False else None)[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
            # Per-team rolling requires a different approach:
            # stack home + away into a long format, then unstack
            df[new_col] = _per_team_rolling(df, col, team_cols, date_col, window)

    return df


def _per_team_rolling(
    df: pd.DataFrame,
    col: str,
    team_cols: tuple[str, str],
    date_col: str,
    window: int,
) -> pd.Series:
    """Compute lag-1 rolling mean for a stat column per team.

    The stat in ``col`` (e.g. ``HomeShots``) is associated with ``team_cols[0]``
    when the team plays at home. We build a unified per-team time series and
    compute the rolling mean over the last ``window`` appearances.
    """
    home_col, away_col = team_cols

    # Build long-format: one row per team per match
    home_df = df[[date_col, home_col, col]].rename(columns={home_col: "_team", col: "_stat"})
    away_col_name = col.replace("Home", "Away")
    if away_col_name in df.columns:
        away_df = df[[date_col, away_col, away_col_name]].rename(
            columns={away_col: "_team", away_col_name: "_stat"}
        )
    else:
        # If no corresponding away column, skip
        return pd.Series([float("nan")] * len(df), index=df.index)

    long_df = pd.concat([home_df.assign(_side="home"), away_df.assign(_side="away")])
    long_df = long_df.sort_values(date_col)

    # Per-team rolling (shift 1 to exclude current match)
    long_df["_rolling"] = long_df.groupby("_team")["_stat"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

    # Re-attach home values to original index
    home_rolling = long_df[long_df["_side"] == "home"].set_index(df.index)["_rolling"]
    return home_rolling
