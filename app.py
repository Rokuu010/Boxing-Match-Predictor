# In this file, I'm building the web application for my Boxing Predictor.
# I'm using the Streamlit library, which makes it fast and easy to create and share

# Path Setup
# I'm adding these lines to manually add my 'src' folder to the Python path.
# Making sure my application can correctly find
# and import the modules from my source library, especially when running scripts
# like this one from the project's root directory.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
# End Path Setup

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# I'm importing all the necessary functions from my 'src' library.
from src import config
from src.data_loader import load_and_clean_data, build_fighters_stats, _resolve_name
from src.predictor import build_feature_row
from src.explain import ensemble_shap_explain, adjust_with_tech_skills
from src.features import feature_cols
from src.utils import get_fighter_data
from src.style import apply_boxing_theme

# Caching
# I'm using Streamlits caching feature to load my model and data only once when the app starts
@st.cache_resource
def load_artifacts():
    """
    Loads the trained model, imputer, dataset, and calculates average stats.
    """
    model = joblib.load(config.MODEL_PATH)
    imputer = joblib.load(config.IMPUTER_PATH)
    df = load_and_clean_data(config.DATA_PATH)
    fighters_stats, all_fighters = build_fighters_stats(df)

    models_for_shap = {
        "XGBoost": joblib.load(config.XGB_PATH),
        "RandomForest": joblib.load(config.RF_PATH)
    }

    # This calculates the average stats for all fighters in my dataset.
    # This will be my fallback for any missing data when I scrape from Wikipedia.
    avg_stats = pd.DataFrame.from_dict(fighters_stats, orient='index').mean().to_dict()

    return model, imputer, fighters_stats, all_fighters, models_for_shap, avg_stats

def get_or_scrape_fighter(name, fighters_stats, all_fighters, avg_stats):
    """
    This function now uses a smarter hybrid approach. It first looks for a
    high-confidence match in the local data. If it doesn't find one, it assumes
    it's a new fighter and scrapes the web for the name the user typed.
    """
    canonical_name = _resolve_name(name, all_fighters)

    if canonical_name:
        base_stats = fighters_stats[canonical_name].copy()
        live_stats = get_fighter_data(canonical_name)
        updates_made = []
        if live_stats.get("age") and live_stats["age"] != base_stats.get("Age"):
            base_stats["Age"] = live_stats["age"]
            updates_made.append(f"age to {live_stats['age']}")
        if live_stats.get("wins") and live_stats["wins"] != base_stats.get("Wins"):
            base_stats["Wins"] = live_stats["wins"]
            updates_made.append(f"wins to {live_stats['wins']}")

        if updates_made:
            st.success(f"Updated live stats for {canonical_name}: {', '.join(updates_made)}.")

        return base_stats, canonical_name, False

    else:
        st.warning(f"'{name}' not found in local data. Attempting to fetch from the web...")
        scraped_stats = get_fighter_data(name)

        ko_perc = None
        if scraped_stats.get("wins") and scraped_stats.get("wins_by_ko"):
            if scraped_stats["wins"] > 0:
                ko_perc = scraped_stats["wins_by_ko"] / scraped_stats["wins"]

        found_stats = []
        if scraped_stats.get("age"): found_stats.append("Age")
        if scraped_stats.get("wins"): found_stats.append("Wins")
        if scraped_stats.get("reach"): found_stats.append("Reach")
        if scraped_stats.get("height"): found_stats.append("Height")
        if ko_perc is not None: found_stats.append("KO %")

        if found_stats:
            st.success(f"Successfully scraped: {', '.join(found_stats)} for '{name}'. Using averages for other metrics.")
        else:
            st.error(f"Could not find any online data for '{name}'. Using full averages.")

        final_stats = {
            "Reach": scraped_stats.get("reach") or avg_stats.get('Reach'),
            "Height": scraped_stats.get("height") or avg_stats.get('Height'),
            "Weight": scraped_stats.get("weight") or avg_stats.get('Weight'),
            "Age": scraped_stats.get("age") or avg_stats.get('Age'),
            "Wins": scraped_stats.get("wins") or avg_stats.get('Wins'),
            "KO%": ko_perc if ko_perc is not None else avg_stats.get('KO%'),
            "RecentWin%": avg_stats.get('RecentWin%'),
            "Defense%": avg_stats.get('Defense%'),
            "PunchAcc": avg_stats.get('PunchAcc'),
            "SoS": avg_stats.get('SoS'),
            "TimeSince": avg_stats.get('TimeSince')
        }

        return final_stats, name, True

# Main Application Logic
def main():
    st.set_page_config(page_title="Boxing Predictor", page_icon="🥊")
    apply_boxing_theme()
    st.title("🥊 Boxing Match Predictor")
    st.write(
        "Welcome to my Boxing Predictor! I built this app to predict the outcome of "
        "fights using a machine learning model. Type in two fighter names below. "
        "If a fighter isn't in my dataset, I'll try to find them on the web."
    )

    try:
        model, imputer, fighters_stats, all_fighters, models_for_shap, avg_stats = load_artifacts()
    except FileNotFoundError:
        st.error("Error: Model files not found. Please run `train.py` first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        fighter_a_name_input = st.text_input("Enter Fighter A", value="")
    with col2:
        fighter_b_name_input = st.text_input("Enter Fighter B", value="")

    if st.button("Predict Winner"):
        if not fighter_a_name_input or not fighter_b_name_input:
            st.warning("Please enter a name for both fighters.")
        elif fighter_a_name_input.lower() == fighter_b_name_input.lower():
            st.warning("Please enter two different fighter names.")
        else:
            with st.spinner(f"Analysing the matchup..."):

                # Build feature row
                stats_a, name_a, is_scraped_a = get_or_scrape_fighter(
                    fighter_a_name_input, fighters_stats, all_fighters, avg_stats
                )
                stats_b, name_b, is_scraped_b = get_or_scrape_fighter(
                    fighter_b_name_input, fighters_stats, all_fighters, avg_stats
                )

                row_df = build_feature_row(stats_a, stats_b)
                row_df = row_df.reindex(columns=feature_cols)
                for col in row_df.columns:
                    if not pd.api.types.is_numeric_dtype(row_df[col]):
                        row_df[col] = pd.to_numeric(row_df[col], errors='coerce')
                row_df_imputed = pd.DataFrame(imputer.transform(row_df), columns=feature_cols)

                contrib_map = ensemble_shap_explain(row_df_imputed, models_for_shap)
                contrib_series = pd.Series(contrib_map)

                proba_A = model.predict_proba(row_df_imputed)[0, 1]
                proba_A_adj = adjust_with_tech_skills(proba_A, contrib_series)

                winner = name_a if proba_A_adj >= 0.5 else name_b
                confidence = float(proba_A_adj if proba_A_adj >= 0.5 else (1 - proba_A_adj))

                st.subheader("Prediction Result")
                st.metric(label=f"Predicted Winner", value=winner, delta=f"Confidence: {confidence:.1%}")

                st.subheader("Tale of the Tape")
                st.write("Here are the key stats my model used for the prediction.")

                stat_display_order = {
                    'Height': 'Height (cm)',
                    'Reach': 'Reach (cm)',
                    'Weight': 'Weight (lbs)',
                    'Age': 'Age',
                    'Wins': 'Wins',
                    'KO%': 'KO %',
                    'SoS': 'Strength of Schedule'
                }

                # Pre-formatting Logic 
                # I'm now preparing the data with formatting applied IN ADVANCE to avoid errors.
                tape_data = {'Stat': list(stat_display_order.values())}

                def format_stats(stats, is_scraped):
                    formatted = []
                    for key, display_name in stat_display_order.items():
                        val = stats.get(key)
                        if pd.isna(val):
                            formatted.append("-")
                        elif key == 'KO%':
                            formatted.append(f"{val:.1%}" + ("*" if is_scraped else ""))
                        elif isinstance(val, float):
                            formatted.append(f"{val:.2f}")
                        else:
                            formatted.append(str(val))
                    return formatted

                tape_data[name_a] = format_stats(stats_a, is_scraped_a)
                tape_data[name_b] = format_stats(stats_b, is_scraped_b)

                df_tape = pd.DataFrame(tape_data).set_index('Stat')

                # Now I can just display the pre-formatted DataFrame
                st.dataframe(df_tape)

                if is_scraped_a or is_scraped_b:
                    st.caption("*Some stats were estimated using a dataset average as they could not be found online.")

                st.subheader("Top Contributing Factors")
                st.write("This chart shows which statistical differences had the biggest impact on the prediction.")

                top_feats = contrib_series.abs().sort_values(ascending=False).head(10).index
                top_contrib = contrib_series[top_feats].sort_values()

                fig, ax = plt.subplots(figsize=(10, 6))
                top_contrib.plot(kind="barh", ax=ax)
                ax.set_title(f"Feature Contributions: {name_a} vs {name_b}")
                ax.set_xlabel("Signed SHAP Contribution (Pushing Prediction)")
                plt.tight_layout()
                st.pyplot(fig)

if __name__ == "__main__":
    main()
