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
from src.utils import get_fighter_data # Import the web scraper

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
    This function now uses a hybrid approach. It tries to find a fighter in my
    local data, and if successful, it then attempts to update their age and wins
    with the latest data scraped from Wikipedia.
    """
    # First, I'll try to find a high-confidence match in my existing dataset.
    canonical_name = _resolve_name(name, all_fighters)

    # If a good match is found, I'll use that fighter's data.
    if canonical_name:
        base_stats = fighters_stats[canonical_name].copy()

        # Now, I perform a live check on the web to get the latest info.
        live_stats = get_fighter_data(canonical_name)
        updates_made = []
        # I only update the age if the scraper finds a new one.
        if live_stats.get("age") and live_stats["age"] != base_stats.get("Age"):
            base_stats["Age"] = live_stats["age"]
            updates_made.append(f"age to {live_stats['age']}")

        # I do the same for the win count.
        if live_stats.get("wins") and live_stats["wins"] != base_stats.get("Wins"):
            base_stats["Wins"] = live_stats["wins"]
            updates_made.append(f"wins to {live_stats['wins']}")

        if updates_made:
            st.success(f"Updated live stats for {canonical_name}: {', '.join(updates_made)}.")
        else:
            st.info(f"Found '{name}' as '{canonical_name}' in the local dataset. Stats are up-to-date.")

        return base_stats, canonical_name, False # Return False as it's not a scraped profile

    # If no high-confidence match was found, I'll treat it as a new fighter.
    else:
        st.warning(f"'{name}' not found in local data. Attempting to fetch from the web...")
        scraped_stats = get_fighter_data(name)

        ko_perc = None
        if scraped_stats.get("wins") and scraped_stats.get("wins_by_ko"):
            if scraped_stats["wins"] > 0:
                ko_perc = scraped_stats["wins_by_ko"] / scraped_stats["wins"]
                st.success(f"Calculated live KO % for {name}: {ko_perc:.1%}")

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
        st.success(f"Successfully scraped basic stats for '{name}'. Using average values for other metrics.")
        return final_stats, name, True # Return True as it IS a scraped profile

# Main Application Logic
def main():
    st.set_page_config(page_title="Boxing Predictor", page_icon="🥊")
    st.title("🥊 Boxing Match Predictor")
    st.write(
        "Welcome to my Boxing Predictor! I built this app to predict the outcome of "
        "fights using a machine learning model. Type in two fighter names below. "
        "If a fighter isn't in my dataset, I'll try to find them on Wikipedia."
    )

    try:
        model, imputer, fighters_stats, all_fighters, models_for_shap, avg_stats = load_artifacts()
    except FileNotFoundError:
        st.error("Error: Model files not found. Please run `train.py` first.")
        return

    # User Interface
    col1, col2 = st.columns(2)
    with col1:
        fighter_a_name_input = st.text_input("Enter Fighter A", value="")
    with col2:
        fighter_b_name_input = st.text_input("Enter Fighter B", value="")

    if st.button("Predict Winner"):
        # I've moved the validation logic here, before any data processing.
        # This logic will now correctly uses the raw string inputs from the text boxes.
        if not fighter_a_name_input or not fighter_b_name_input:
            st.warning("Please enter a name for both fighters.")
        elif fighter_a_name_input.lower() == fighter_b_name_input.lower():
            st.warning("Please enter two different fighter names.")
        else:
            with st.spinner(f"Analysing the matchup..."):
                # Now that the inputs are validated, I can safely fetch the stats.
                stats_a, name_a, is_scraped_a = get_or_scrape_fighter(fighter_a_name_input, fighters_stats, all_fighters, avg_stats)
                stats_b, name_b, is_scraped_b = get_or_scrape_fighter(fighter_b_name_input, fighters_stats, all_fighters, avg_stats)

                # Prediction and Explanation Logic
                row_df = build_feature_row(stats_a, stats_b)
                row_df_imputed = pd.DataFrame(imputer.transform(row_df), columns=feature_cols)

                contrib_map = ensemble_shap_explain(row_df_imputed, models_for_shap)
                contrib_series = pd.Series(contrib_map)

                proba_A = model.predict_proba(row_df_imputed)[0, 1]
                proba_A_adj = adjust_with_tech_skills(proba_A, contrib_series)

                winner = name_a if proba_A_adj >= 0.5 else name_b
                confidence = float(proba_A_adj if proba_A_adj >= 0.5 else (1 - proba_A_adj))

                # Displaying the Results
                st.subheader("Prediction Result")
                st.metric(label=f"Predicted Winner", value=winner, delta=f"Confidence: {confidence:.1%}")

                st.subheader("Tale of the Tape")
                st.write("Here are the key stats my model used for the prediction.")

                stat_display_order = {
                    'Height': 'Height (cm)', 'Reach': 'Reach (cm)', 'Weight': 'Weight (lbs)',
                    'Age': 'Age', 'Wins': 'Wins', 'KO%': 'KO %', 'SoS': 'Strength of Schedule'
                }
                
                tape_data = {'Stat': list(stat_display_order.values())}
                tape_data[name_a] = [stats_a.get(key) for key in stat_display_order.keys()]
                tape_data[name_b] = [stats_b.get(key) for key in stat_display_order.keys()]

                ko_index = list(stat_display_order.keys()).index('KO%')
                for name, is_scraped in [(name_a, is_scraped_a), (name_b, is_scraped_b)]:
                    ko_val = tape_data[name][ko_index]
                    if pd.notna(ko_val):
                        tape_data[name][ko_index] = f"{ko_val:.1%}" + ("*" if is_scraped else "")

                df_tape = pd.DataFrame(tape_data).set_index('Stat')
                st.dataframe(df_tape.style.format("{:.2f}", na_rep="-"))

                if is_scraped_a or is_scraped_b:
                    st.caption("*Stat is an estimated average from my dataset as it could not be found online.")

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

