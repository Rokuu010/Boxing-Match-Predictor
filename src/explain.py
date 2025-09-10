# src/explain.py

# This is the explainability hub of my project. I built these functions to look inside
# the "black box" of my model and understand *why* it makes a certain prediction.
# My main goal was to use the SHAP library to get detailed, feature-by-feature
# contributions for any given fight matchup.

import os
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

from . import config
# I've updated the imports to break the circular dependency.
# Data utilities now come from data_loader, and the feature row builder from predictor.
from .data_loader import _resolve_name, build_fighters_stats
from .predictor import build_feature_row
from .features import feature_cols # I'm also importing the feature_cols list

# I'm using a try/except block to import SHAP. This makes the library an optional
# dependency, meaning my core prediction code will still run even if SHAP isn't installed.
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# --- SHAP Calculation Helpers ---
# I created these helper functions to handle the complexities of calculating SHAP values
# for the different types of models within my ensemble.

def _tree_shap_values(model, row_df):
    """
    A dedicated function to get SHAP values specifically from tree-based models
    like Random Forest and XGBoost. I added logic to handle different output
    formats from various versions of the SHAP library to make my code more robust.
    """
    expl = shap.TreeExplainer(model)
    vals = expl.shap_values(row_df)
    vals = np.array(vals)

    # I handle various SHAP output shapes to ensure consistency.
    if vals.ndim == 2 and vals.shape[0] == 1:
        vals = vals[0]
    if vals.ndim == 3: # For multi-class outputs, I take the values for the "win" class.
        vals = vals[:, :, 1].mean(axis=0)
    if vals.ndim == 2 and vals.shape[0] > 1:
        vals = vals.mean(axis=0)
    return vals

def ensemble_shap_explain(row_df, models: dict) -> dict:
    """
    This function calculates the SHAP values for each model in my ensemble
    (XGBoost and Random Forest) and then averages them. This gives me a single,
    unified explanation for the ensemble's overall prediction.
    """
    agg = np.zeros(len(feature_cols), dtype=float)
    names = list(models.keys())

    for name in names:
        model = models[name]
        try:
            if HAS_SHAP:
                contrib = _tree_shap_values(model, row_df)
            else: # If SHAP isn't available, I just use zero contribution.
                contrib = np.zeros(len(feature_cols), dtype=float)
        except Exception as e:
            logging.warning(f"SHAP calculation failed for {name}: {e}. Using zeros.")
            contrib = np.zeros(len(feature_cols), dtype=float)
        agg += contrib / len(names) # Simple average of contributions.
    return dict(zip(feature_cols, agg))

def adjust_with_tech_skills(proba_A, contrib_series):
    """
    This is a custom rule I created. I felt the model might sometimes overlook
    key technical skills, so I designed this function to make a small adjustment
    to the final probability based on the SHAP contributions of features like
    'PunchAccDiff', 'DefenseDiff', and 'ReachDiff'.
    """
    tech_feats = ["PunchAccDiff", "DefenseDiff", "ReachDiff"]
    tech_score = sum(contrib_series.get(f, 0) for f in tech_feats)
    # I use a tanh function to make sure the adjustment is always small and smooth.
    adjustment = 0.1 * np.tanh(tech_score / 5.0)
    adjusted_proba = proba_A + adjustment
    # I clamp the final value between 0 and 1 to ensure it's a valid probability.
    return min(max(adjusted_proba, 0), 1)

def explain_fight(fighterA: str,
                  fighterB: str,
                  model,
                  imputer,
                  df):
    """
    This is my main function for explaining a single fight prediction.

    It orchestrates the entire process: preparing the data for a single matchup,
    calculating the feature contributions using SHAP, adjusting the final probability,
    and then presenting the results in a clear and understandable way (in the console,
    as a plot, and saved to files).
    """
    out_dir = config.EXPLANATION_DIR
    top_k = 10
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Data Preparation ---
    fighters_stats, all_fighters = build_fighters_stats(df)
    A = _resolve_name(fighterA, all_fighters)
    B = _resolve_name(fighterB, all_fighters)

    stats_a = fighters_stats.get(A)
    stats_b = fighters_stats.get(B)

    # This now uses the corrected function name and arguments.
    row_df = build_feature_row(stats_a, stats_b)
    if imputer is not None:
        row_df = pd.DataFrame(imputer.transform(row_df), columns=feature_cols)

    # --- 2. SHAP Contribution Calculation ---
    models_for_shap = {}
    try:
        if os.path.exists(config.XGB_PATH):
            models_for_shap["XGBoost"] = joblib.load(config.XGB_PATH)
        if os.path.exists(config.RF_PATH):
            models_for_shap["RandomForest"] = joblib.load(config.RF_PATH)
    except Exception as e:
        logging.warning(f"Could not load sub-models for SHAP: {e}")

    contrib_series = None
    method_label = "Unknown"

    if HAS_SHAP and models_for_shap:
        try:
            contrib_map = ensemble_shap_explain(row_df, models_for_shap)
            contrib_series = pd.Series(contrib_map)
            method_label = "SHAP (avg RF + XGB)"
        except Exception as e:
            logging.warning(f"SHAP explain failed: {e}")

    if contrib_series is None:
        try:
            rf = models_for_shap.get("RandomForest")
            if rf is not None:
                importances = pd.Series(rf.feature_importances_, index=feature_cols)
                feature_values = row_df.iloc[0]
                contrib_series = feature_values * importances
                method_label = "RF importances × diffs (fallback)"
            else:
                raise RuntimeError("Random Forest model not available for fallback.")
        except Exception as e:
            logging.warning(f"Fallback explanation failed: {e}")
            contrib_series = pd.Series(0.0, index=feature_cols)

    # --- 3. Final Prediction and Output ---
    proba_A = model.predict_proba(row_df)[0, 1]
    proba_A_adj = adjust_with_tech_skills(proba_A, contrib_series)

    winner = A if proba_A_adj >= 0.5 else B
    confidence = float(proba_A_adj if proba_A_adj >= 0.5 else (1 - proba_A_adj))

    top_feats = contrib_series.abs().sort_values(ascending=False).head(top_k).index
    top_contrib = contrib_series[top_feats].sort_values()

    print(f"\n🥊 {A} vs {B}")
    print(f"Predicted winner: {winner} (Confidence: {confidence:.2%})")
    print(f"Explanation method: {method_label}")
    print("\nTop contributing factors (positive → pushes toward A; negative → toward B):")
    for feat in top_feats:
        print(f" - {feat}: value={row_df.iloc[0][feat]:.4f}, contribution={contrib_series[feat]:.4f}")

    plt.figure(figsize=(8, 6))
    top_contrib.plot(kind="barh")
    plt.title(f"Feature Contributions: {A} vs {B}")
    plt.xlabel("Signed Contribution (Pushing Prediction)")
    plt.tight_layout()
    plt.show()

    base = f"{A.replace(' ','_')}_vs_{B.replace(' ','_')}"
    csv_path = os.path.join(out_dir, f"{base}_explanation.csv")
    json_path = os.path.join(out_dir, f"{base}_explanation.json")

    rows = [{"feature": f, "value": float(row_df.iloc[0][f]), "contribution": float(contrib_series[f])} for f in top_feats]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    record = {
        "fighter_A": A, "fighter_B": B,
        "predicted_winner": winner, "confidence": confidence,
        "raw_probability_A": float(proba_A),
        "adjusted_probability_A": float(proba_A_adj),
        "explanation_method": method_label,
        "top_contributions": rows
    }
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\nSaved explanation to: {csv_path}")