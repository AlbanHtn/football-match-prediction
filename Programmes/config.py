import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR_E = BASE_DIR.parent / "Donnees"
DATA_DIR_S = BASE_DIR.parent.parent / "Donnees"

DATA_DIR = DATA_DIR_E

FILE_MATCHES = DATA_DIR_E / "Matches.csv"
FILE_ELO = DATA_DIR_E / "EloRatings.csv"

FILE_MAPPING_SUGGEST = DATA_DIR_E / "club_mapping_suggestions.csv"
FILE_MAPPING_CLEAN = DATA_DIR_E / "club_mapping_clean.csv"
FILE_MERGE_COUNTRY = DATA_DIR_E /"division_to_country_TEMPLATE.csv"

FILE_MATCHES_ELO = DATA_DIR_S / "Matches_with_Elo.csv"        
FILE_CLEAN = DATA_DIR_S / "matches_clean.csv"                 
FILE_RECENT = DATA_DIR_S / "matches_recent.csv"               
FILE_PRED_FULL = DATA_DIR_S / "matches_pred_full.csv"         
FILE_SELECT = DATA_DIR_S / "matches_select_championnats.csv"  
