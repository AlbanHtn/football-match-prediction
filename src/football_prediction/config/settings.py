"""
Centralized configuration for the football prediction pipeline.

All file paths resolve relative to the repository root so they work
on any machine regardless of where the repo is cloned.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Repository root: two levels up from this file (src/football_prediction/config/)
REPO_ROOT = Path(__file__).parents[3].resolve()


@dataclass(frozen=True)
class DataPaths:
    """Canonical paths for every data artifact in the pipeline."""

    # ── Repository layout ──────────────────────────────────────────────────
    repo_root: Path = field(default_factory=lambda: REPO_ROOT)

    # ── Raw sources (large — gitignored, must be placed manually) ─────────
    raw_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "raw")

    @property
    def raw_matches(self) -> Path:
        return self.raw_dir / "Matches.csv"

    @property
    def raw_elo(self) -> Path:
        return self.raw_dir / "EloRatings.csv"

    # ── Reference tables (small — versioned in git) ───────────────────────
    references_dir: Path = field(default_factory=lambda: REPO_ROOT / "references")

    @property
    def club_mapping_clean(self) -> Path:
        return self.references_dir / "club_mapping_clean.csv"

    @property
    def club_mapping_suggestions(self) -> Path:
        return self.references_dir / "club_mapping_suggestions.csv"

    @property
    def division_to_country(self) -> Path:
        return self.references_dir / "division_to_country.csv"

    # ── Intermediate / processed outputs (gitignored) ─────────────────────
    interim_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "interim")
    processed_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "processed")

    @property
    def matches_with_elo(self) -> Path:
        """Full merged dataset with Elo ratings (165k rows)."""
        return self.interim_dir / "Matches_with_Elo.csv"

    @property
    def matches_pred_full(self) -> Path:
        """Pre-match features only, all leagues (165k rows)."""
        return self.processed_dir / "matches_pred_full.csv"

    @property
    def matches_recent(self) -> Path:
        """Post-2010 matches with sufficient Elo coverage (138k rows)."""
        return self.processed_dir / "matches_recent.csv"

    @property
    def matches_clean(self) -> Path:
        """High-completeness subset — all features available (8.5k rows)."""
        return self.processed_dir / "matches_clean.csv"

    @property
    def matches_big5(self) -> Path:
        """Big 5 leagues selection (52k rows)."""
        return self.processed_dir / "matches_select_championnats.csv"

    # ── Legacy paths (Donnees/ directory) — kept for backward compat ──────
    legacy_dir: Path = field(default_factory=lambda: REPO_ROOT / "Donnees")

    @property
    def legacy_matches(self) -> Path:
        return self.legacy_dir / "Matches.csv"

    @property
    def legacy_elo(self) -> Path:
        return self.legacy_dir / "EloRatings.csv"


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters and modelling constants."""

    # ── Dataset filters ────────────────────────────────────────────────────
    big5_divisions: tuple[str, ...] = ("E0", "F1", "D1", "I1", "SP1")
    min_season_year: int = 2006          # Matches before this year excluded
    completeness_threshold: float = 0.80 # Minimum non-null ratio per column

    # ── Cross-validation ───────────────────────────────────────────────────
    n_cv_splits: int = 5

    # ── Target ────────────────────────────────────────────────────────────
    target_column: str = "FTResult"
    target_classes: tuple[str, ...] = ("H", "D", "A")

    # ── Sample weighting ──────────────────────────────────────────────────
    draw_sample_weight: float = 1.1     # Slight upweight to counter draw under-prediction

    # ── Rolling window sizes ───────────────────────────────────────────────
    rolling_windows: tuple[int, ...] = (3, 5)

    # ── Features — pre-match only (no data leakage) ───────────────────────
    macro_features: tuple[str, ...] = (
        "HomeElo", "AwayElo", "EloDiff",
        "Form3Home", "Form3Away",
        "FormMomentumHome", "FormMomentumAway",
    )
    odds_features: tuple[str, ...] = (
        "OddHome", "OddDraw", "OddAway",
        "ImpliedProbHome", "ImpliedProbDraw", "ImpliedProbAway",
        "BookmakerMargin",
    )


# ── Singleton instances ────────────────────────────────────────────────────────
settings = DataPaths()
model_config = ModelConfig()
