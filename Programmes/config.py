"""Backward-compatible config shim for notebooks in Programmes/.

The canonical configuration now lives in ``football_prediction.config.settings``.
This module re-exports the legacy path constants used by existing notebooks so
they keep working without modification, while new code should import from the
package directly:

    from football_prediction.config.settings import DataPaths, ModelConfig
    paths = DataPaths()
"""

from football_prediction.config.settings import DataPaths

_paths = DataPaths()

# ── Directories ────────────────────────────────────────────────────────────────
BASE_DIR = _paths.repo_root / "Programmes"
DATA_DIR_E = _paths.legacy_dir          # /Donnees
DATA_DIR_S = _paths.legacy_dir          # historically a sibling — now unified
DATA_DIR = DATA_DIR_E

# ── Raw sources ────────────────────────────────────────────────────────────────
FILE_MATCHES = _paths.legacy_matches
FILE_ELO = _paths.legacy_elo

# ── Reference tables ──────────────────────────────────────────────────────────
FILE_MAPPING_SUGGEST = DATA_DIR_E / "club_mapping_suggestions.csv"
FILE_MAPPING_CLEAN = DATA_DIR_E / "club_mapping_clean.csv"
FILE_MERGE_COUNTRY = DATA_DIR_E / "division_to_country_TEMPLATE.csv"

# ── Intermediate / processed outputs ──────────────────────────────────────────
FILE_MATCHES_ELO = DATA_DIR_S / "Matches_with_Elo.csv"
FILE_CLEAN = DATA_DIR_S / "matches_clean.csv"
FILE_RECENT = DATA_DIR_S / "matches_recent.csv"
FILE_PRED_FULL = DATA_DIR_S / "matches_pred_full.csv"
FILE_SELECT = DATA_DIR_S / "matches_select_championnats.csv"
