"""Raw data loading with schema validation."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Expected dtypes for schema-level validation at load time
_MATCHES_PARSE_DATES = ["MatchDate"]
_ELO_PARSE_DATES = ["Date"]


def _detect_sep(path: Path) -> str:
    """Detect whether a CSV uses ',' or ';' by inspecting its header line.

    The raw sources (Football-Data.co.uk, ClubElo) are semicolon-delimited;
    comma is kept as the fallback for already-normalized inputs (e.g. tests).
    """
    with path.open(encoding="utf-8") as f:
        header = f.readline()
    return ";" if header.count(";") > header.count(",") else ","


def load_matches(path: Path) -> pd.DataFrame:
    """Load raw Matches.csv with correct types.

    Args:
        path: Absolute path to Matches.csv.

    Returns:
        DataFrame with MatchDate parsed as datetime.

    Raises:
        FileNotFoundError: If the file does not exist at ``path``.
        ValueError: If required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Matches source file not found: {path}\n"
            "Place Matches.csv in data/raw/ (see README for download instructions)."
        )

    logger.info("Loading matches from %s", path)
    df = pd.read_csv(path, sep=_detect_sep(path), low_memory=False)

    required = {"Division", "MatchDate", "HomeTeam", "AwayTeam", "FTResult"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Matches CSV is missing required columns: {missing}")

    df["MatchDate"] = pd.to_datetime(df["MatchDate"], dayfirst=True)

    logger.info("Loaded %d matches across %d divisions", len(df), df["Division"].nunique())
    return df


def load_elo(path: Path) -> pd.DataFrame:
    """Load raw EloRatings.csv with correct types.

    Args:
        path: Absolute path to EloRatings.csv.

    Returns:
        DataFrame with Date parsed as datetime.

    Raises:
        FileNotFoundError: If the file does not exist at ``path``.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Elo ratings file not found: {path}\n"
            "Place EloRatings.csv in data/raw/ (see README for download instructions)."
        )

    logger.info("Loading Elo ratings from %s", path)
    df = pd.read_csv(path, sep=_detect_sep(path))
    canonical = {"date": "Date", "club": "Club", "elo": "Elo"}
    df = df.rename(columns={c: canonical[c.lower()] for c in df.columns if c.lower() in canonical})

    required = {"Date", "Club", "Elo"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"EloRatings CSV is missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    logger.info("Loaded %d Elo snapshots for %d clubs", len(df), df["Club"].nunique())
    return df
