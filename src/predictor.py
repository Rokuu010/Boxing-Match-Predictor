# In this file, I've centralised all the logic related to making a single prediction.
# This includes functions for preparing the data for a matchup and calling the
# trained model to get the final prediction probability.

import pandas as pd
from .data_loader import _resolve_name, build_fighters_stats
from .features import feature_cols

def build_feature_row(statsA, statsB):
    """
    Once I have the stats for two fighters, this function takes those stats
    and calculates the engineered features (e.g., 'ReachDiff', 'KODiff') needed
    by my model. It then returns them as a single-row DataFrame.
    """
    # This dictionary directly mirrors the feature engineering logic in my features.py.
    feats = {
        "ReachDiff": statsA.get("Reach", 0) - statsB.get("Reach", 0),
        "AgeDiff": statsA.get("Age", 0) - statsB.get("Age", 0),
        "HeightDiff": statsA.get("Height", 0) - statsB.get("Height", 0),
        "WeightDiff": statsA.get("Weight", 0) - statsB.get("Weight", 0),
        "KODiff": statsA.get("KO%", 0) - statsB.get("KO%", 0),
        "WinDiff": statsA.get("Wins", 0) - statsB.get("Wins", 0),
        "RecentWinPercDiff": statsA.get("RecentWin%", 0) - statsB.get("RecentWin%", 0),
        "DefenseDiff": statsA.get("Defense%", 0) - statsB.get("Defense%", 0),
        "PunchAccDiff": statsA.get("PunchAcc", 0) - statsB.get("PunchAcc", 0),
        "SoSDiff": statsA.get("SoS", 0) - statsB.get("SoS", 0),
        "TimeSinceDiff": statsA.get("TimeSince", 0) - statsB.get("TimeSince", 0)
    }
    feats["PowerVsSchedule"] = feats.get("KODiff", 0) * feats.get("SoSDiff", 0)
    feats["WinsVsSchedule"] = feats.get("WinDiff", 0) * feats.get("SoSDiff", 0)

    sos_diff = feats.get("SoSDiff", 0)
    recent_win_diff = feats.get("RecentWinPercDiff", 0)
    feats["RecentFormWeighted"] = recent_win_diff * (1 + 0.5 * (sos_diff / (1 + abs(sos_diff))))

    row = pd.DataFrame([feats])
    # I make sure the columns are in the exact order the model expects.
    return row[feature_cols]


def predict_fight(fighterA, fighterB, model, imputer, df, return_dict=True):
    """
    This is my main prediction function that ties everything together.
    It takes the fighter names, orchestrates the data preparation and feature
    engineering, and then uses the trained model and imputer to get the final result.
    """
    fighters_stats, all_fighters = build_fighters_stats(df)
    A = _resolve_name(fighterA, all_fighters)
    B = _resolve_name(fighterB, all_fighters)

    X_pred = build_feature_row(fighters_stats[A], fighters_stats[B])
    if imputer is not None:
        X_pred = pd.DataFrame(imputer.transform(X_pred), columns=feature_cols)

    probs = model.predict_proba(X_pred)[0]
    proba_A = probs[1]

    pred_A_wins = proba_A >= 0.5
    winner = A if pred_A_wins else B
    confidence = float(proba_A if pred_A_wins else (1 - proba_A))

    if return_dict:
        return {"fighter_A": A, "fighter_B": B, "winner": winner, "confidence": confidence, "proba_A": proba_A}

    return winner, confidence