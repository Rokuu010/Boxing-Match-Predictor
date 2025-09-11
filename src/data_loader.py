# src/data_loader.py

# At this point I've centralised all the functions related to loading and preparing
# my initial dataset. I also decided to place my common helper functions for finding
# fighter names and building the stats dictionary here. Since both my prediction and
# explanation scripts need these functions, putting them in this lower-level
# was the best way to prevent circular import errors and keep my code organised.

import pandas as pd
import logging
import difflib
from . import config

# Fighter Name Resolution & Stats Building
# These are the shared utility functions I moved here from the predictor script.

# I created this alias map as a way to manually correct any recurring or
# tricky name mismatches, for example, if a fighter uses a nickname.
alias_map = {}

def register_alias(alias, canonical):
    """A helper function to add new aliases to the map programmatically."""
    alias_map[alias] = canonical

def _resolve_name(name, choices, cutoff=0.95):
    """
    I designed this function to reliably find a fighter's name from user input.
    It's crucial for making my interactive app user-friendly. My process is:
    1. Check my custom alias map first for any manual overrides.
    2. Look for an exact match in the list of known fighters.
    3. If no exact match is found, I use a fuzzy string match with a high
       confidence threshold (95%) to correct minor spelling mistakes.
    4. As a final fallback, I try again with a lower threshold to catch more
       significant typos. If it's still not found, I raise an error.
    """
    if name in alias_map:
        return alias_map[name]
    if name in choices:
        return name

    # First attempt: high confidence fuzzy match
    cand = difflib.get_close_matches(name, choices, n=1, cutoff=cutoff)
    if cand:
        return cand[0]

    # Second attempt: lower confidence fallback
    cand2 = difflib.get_close_matches(name, choices, n=1, cutoff=0.6)
    if cand2:
        logging.info(f"Resolved '{name}' to '{cand2[0]}' with low confidence.")
        return cand2[0]

    raise ValueError(f"Fighter '{name}' could not be found or matched.")

def build_fighters_stats(df):
    """
    This function processes my main dataframe and creates a clean dictionary where
    each fighter's name maps to their latest statistics. This is much more
    efficient than searching the whole dataframe every time I need to look up a fighter.
    """
    fighters_stats = {}
    # I iterate through each fight record in the dataset.
    for _, row in df.iterrows():
        # For each fight, I add or update the stats for both Fighter A and Fighter B.
        # If a fighter appears multiple times, this logic means their stats from their
        # most recent fight in the dataset will be the ones that are kept.
        if pd.notna(row.get("FighterA")):
            fighters_stats[row["FighterA"]] = {
                "Reach": row.get("ReachA"), "Age": row.get("AgeA"),
                "Height": row.get("HeightA"), "Weight": row.get("WeightA"),
                "KO%": row.get("KOPercA"), "Wins": row.get("WinsA"),
                "RecentWin%": row.get("RecentWinPercA"), "Defense%": row.get("DefensePercA"),
                "PunchAcc": row.get("PunchAccuracyA"), "SoS": row.get("StrengthOfScheduleA"),
                "TimeSince": row.get("TimeSinceLastFightA")
            }
        if pd.notna(row.get("FighterB")):
            fighters_stats[row["FighterB"]] = {
                "Reach": row.get("ReachB"), "Age": row.get("AgeB"),
                "Height": row.get("HeightB"), "Weight": row.get("WeightB"),
                "KO%": row.get("KOPercB"), "Wins": row.get("WinsB"),
                "RecentWin%": row.get("RecentWinPercB"), "Defense%": row.get("DefensePercB"),
                "PunchAcc": row.get("PunchAccuracyB"), "SoS": row.get("StrengthOfScheduleB"),
                "TimeSince": row.get("TimeSinceLastFightB")
            }
    all_fighters = sorted(fighters_stats.keys())
    return fighters_stats, all_fighters


def load_and_clean_data(csv_path=None):
    """
    Loads my boxing dataset from the specified CSV file and performs initial cleaning.

    My cleaning process involves:
    - Converting the 'Result' column to a binary format (1 for a Fighter A win, 0 for anything else).
    - Ensuring all the numeric columns I need for my features are correctly typed as numbers.
    - Dropping any rows where the fight 'Result' is missing, as they are unusable for training.
    """
    if csv_path is None:
        csv_path = config.DATA_PATH

    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Dataset loaded from {csv_path} with {df.shape[0]} rows.")
    except FileNotFoundError:
        logging.error(f"Fatal: Data file not found at {csv_path}")
        raise

    # I'm converting the target variable to a simple binary format.
    df["Result"] = df["Result"].apply(lambda x: 1 if x == 1 else 0)

    # I explicitly define all the columns that should be numeric. This helps catch
    # any data entry errors and ensures they are ready for mathematical operations.
    numeric_cols = [
        "ReachA","ReachB","HeightA","HeightB","WeightA","WeightB",
        "AgeA","AgeB","WinsA","WinsB","KOPercA","KOPercB",
        "RecentWinPercA","RecentWinPercB","DefensePercA","DefensePercB",
        "PunchAccuracyA","PunchAccuracyB","StrengthOfScheduleA","StrengthOfScheduleB",
        "TimeSinceLastFightA","TimeSinceLastFightB"
    ]
    # 'errors="coerce"' is a useful setting that will turn any non-numeric values into 'NaN'.
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Data Sanitisation Step
    # I noticed some KO percentages might be entered as whole numbers (e.g., 88)
    # instead of decimals (0.88). This code finds any values over 1.0 and
    # divides them by 100 to correct them. This prevents impossible stats like a 121% KO average.
    for col in ["KOPercA", "KOPercB"]:
        # I'm applying this correction to any row where the value is greater than 1.
        df.loc[df[col] > 1, col] = df.loc[df[col] > 1, col] / 100
        logging.info(f"Sanitised KO percentages in column: {col}")

    df = df.dropna(subset=["Result"])

    logging.info(f"Data cleaned and sanitised. {df.shape[0]} rows remaining.")
    return df