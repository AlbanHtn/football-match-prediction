"""Dataset filtering and cleaning utilities."""

import logging
from collections.abc import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

# Big 5 European leagues — canonical division codes used in Football-Data.co.uk
BIG5_DIVISIONS = ("E0", "F1", "D1", "I1", "SP1")


def filter_big5(df: pd.DataFrame, division_col: str = "Division") -> pd.DataFrame:
    """Keep only Big 5 European league matches.

    Args:
        df: Input DataFrame with a division column.
        division_col: Name of the division identifier column.

    Returns:
        Filtered DataFrame.
    """
    mask = df[division_col].isin(BIG5_DIVISIONS)
    result = df[mask].copy()
    logger.info(
        "Big5 filter: %d → %d rows (removed %d)",
        len(df), len(result), len(df) - len(result),
    )
    return result


def filter_post_year(
    df: pd.DataFrame,
    min_year: int,
    date_col: str = "MatchDate",
) -> pd.DataFrame:
    """Keep only matches from ``min_year`` onwards.

    Args:
        df: Input DataFrame.
        min_year: Minimum season start year (inclusive).
        date_col: Name of the datetime column.

    Returns:
        Filtered DataFrame.
    """
    mask = df[date_col].dt.year >= min_year
    result = df[mask].copy()
    logger.info(
        "Year filter (>= %d): %d → %d rows",
        min_year, len(df), len(result),
    )
    return result


def apply_completeness_filter(
    df: pd.DataFrame,
    columns: Iterable[str],
    threshold: float = 0.80,
) -> pd.DataFrame:
    """Remove rows where any of the specified columns exceed the null threshold.

    Args:
        df: Input DataFrame.
        columns: Column names to check for completeness.
        threshold: Minimum fraction of non-null values required across the
            full dataset (used only for logging). Row-level: drop any row
            where at least one of ``columns`` is null.

    Returns:
        DataFrame with incomplete rows removed.
    """
    cols = list(columns)
    before = len(df)
    result = df.dropna(subset=cols).copy()
    completeness = result[cols].notna().mean()

    logger.info(
        "Completeness filter on %d columns: %d → %d rows (threshold=%.0f%%)",
        len(cols), before, len(result), threshold * 100,
    )
    logger.debug("Per-column completeness:\n%s", completeness.to_string())
    return result
