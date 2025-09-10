# src/config.py
import os

# I created this configuration file to centralise all the important, hard-coded
# variables for my project. This makes the main scripts cleaner and allows me
# to change key parameters easily from one location.

# --- File Paths ---
# I've organised all my file paths in this section.

# This constant points to my main dataset.
DATA_PATH = "data/boxing_data.csv"

# I'm defining a single directory to store all my trained model files and other artifacts.
MODELS_DIR = "models"
# This line makes sure the models directory is created automatically if it doesn't exist.
os.makedirs(MODELS_DIR, exist_ok=True)

# These are the filepaths for my final, production-ready model and the imputer.
# The .pkl extension is standard for saving Python objects using joblib
MODEL_PATH = os.path.join(MODELS_DIR, "fight_predictor_ensemble.pkl")
IMPUTER_PATH = os.path.join(MODELS_DIR, "imputer.pkl")

# I also save the individual tuned models. I do this so I can load them later
# in my explain.py script to generate detailed SHAP values.
RF_PATH = os.path.join(MODELS_DIR, "rf_best.pkl")
XGB_PATH = os.path.join(MODELS_DIR, "xgb_best.pkl")

# I've set up a dedicated directory for saving the prediction explanation charts and data.
EXPLANATION_DIR = "explanations"


# --- Model Training Hyperparameters ---
# Here, I've defined the key numbers that control the model training process.

# I've set aside 20% of my data as a hold-out test set to evaluate the final model.
TEST_SIZE = 0.2

# I use a fixed random state throughout the project.
# This ensures that every time I run my code, any 'random' processes like data splits
# and model initialisations are the same, making my results reproducible.
RANDOM_STATE = 42

# I'm using 5-fold cross-validation during the hyperparameter tuning phase.
CV_FOLDS = 5

# For my RandomizedSearchCV, I've set it to try 30 different parameter combinations.
# This provides a good balance between finding strong hyperparameters and keeping the
# training time reasonable.
N_ITER_SEARCH = 30

