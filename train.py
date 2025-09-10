# In this script, I've centralised the entire model training process.
# Its only job is to run the full pipeline—from loading data to training,
# tuning, and finally saving all the model files (like the .pkl files)
# to the 'models' directory. I run this script whenever I need to retrain
# the model, for example, after updating my dataset or feature engineering logic.

# --- Path Setup ---
# I'm adding this to ensure the script can correctly find and import my 'src' library.
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
# --- End Path Setup ---

import logging
from src import config
from src.utils import setup_logging
from src.data_loader import load_and_clean_data
from src.features import engineer_features, feature_cols
from src.model import train_model

def main():
    """
    Main function to execute the model training pipeline.
    """
    setup_logging()
    logging.info("--- Starting Model Training Pipeline ---")

    # 1. Load and process the data
    df = load_and_clean_data(config.DATA_PATH)
    df, _ = engineer_features(df) # We get the feature columns from the features module directly

    # 2. Train the model
    # This function handles the entire training, tuning, and evaluation process,
    # and it also saves the final model and imputer to disk, as defined in the config.
    train_model(df, feature_cols)

    logging.info("--- Model Training Pipeline Complete ---")
    logging.info(f"All model files have been saved to the '{config.MODELS_DIR}' directory.")

if __name__ == "__main__":
    main()