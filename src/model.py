# src/model.py
import pandas as pd
import numpy as np
import joblib
import logging

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    balanced_accuracy_score, f1_score, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from . import config


def train_model(df, feature_cols, calibrate=True, test_size=None, random_state=None):
    """
    Trains, tunes, and evaluates the complete machine learning model pipeline.

    This function performs the following steps:
    1. Preprocesses the data by imputing missing values.
    2. Splits data into training and testing sets.
    3. Handles class imbalance in the training set using SMOTE.
    4. Tunes hyperparameters for Random Forest and XGBoost models.
    5. Assembles a soft-voting ensemble model.
    6. Calibrates the model's probabilities.
    7. Evaluates the final model and saves the trained model.

    Returns:
        A tuple containing the final model, imputer, and test data (X_test, y_test).
    """
    # Use default values from the config file if none are provided.
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE

    # == 1. Data Preparation ==
    # Separate features (X) and target variable (y).
    X = df[feature_cols].copy()
    y = df["Result"].astype(int).copy()

    # Fill any missing values in the feature set using the mean of each column.
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

    # Split the data into a training set and a hold-out test set.
    # 'stratify=y' ensures both sets have a similar proportion of wins/losses.
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # == 2. Handling Class Imbalance ==
    # Use SMOTE to create synthetic examples of the minority class.
    # This is done *only* on the training data to prevent data leakage into the test set.
    smote = SMOTE(random_state=random_state, k_neighbors=1)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logging.info(f"Training set size after SMOTE: {X_train_res.shape[0]} samples")

    # Prepare for cross-validation during tuning.
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=random_state)

    # == 3. Hyperparameter Tuning ==
    # Use RandomizedSearchCV to efficiently find the best settings for our models
    # without exhaustively trying every single combination.

    # Tune the Random Forest model.
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rf_dist = {
        "n_estimators":  [200, 300, 500, 700, 900],
        "max_depth":     [6, 10, 14, 18, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":  ["sqrt", "log2", None]
    }
    rf_search = RandomizedSearchCV(
        rf, rf_dist, n_iter=config.N_ITER_SEARCH, scoring="roc_auc", cv=cv, n_jobs=-1, random_state=random_state
    )
    rf_search.fit(X_train_res, y_train_res)
    rf_best = rf_search.best_estimator_
    logging.info(f"Best Random Forest params: {rf_search.best_params_}")

    # Tune the XGBoost model.
    # NOTE: I've removed `use_label_encoder=False` as it is outdated and causes a warning.
    xgb_base = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=random_state,
        tree_method="hist", n_jobs=-1
    )
    xgb_dist = {
        "n_estimators": [300, 500, 700, 900],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7]
    }
    xgb_search = RandomizedSearchCV(
        xgb_base, xgb_dist, n_iter=config.N_ITER_SEARCH, scoring="roc_auc", cv=cv, n_jobs=-1, random_state=random_state
    )
    xgb_search.fit(X_train_res, y_train_res)
    xgb_best = xgb_search.best_estimator_
    logging.info(f"Best XGBoost params: {xgb_search.best_params_}")

    # == 4. Model Assembly and Final Training ==
    # Create a simple Logistic Regression model to add diversity to the ensemble.
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")

    # Combine the three tuned models into a single, more powerful ensemble.
    # 'voting="soft"' averages their prediction probabilities, which is generally more effective.
    ensemble = VotingClassifier(
        estimators=[("rf", rf_best), ("xgb", xgb_best), ("lr", logreg)],
        voting="soft",
        n_jobs=-1
    )
    ensemble.fit(X_train_res, y_train_res)

    # Calibrate the ensemble to make its confidence scores more reliable.
    final_model = ensemble
    if calibrate:
        final_model = CalibratedClassifierCV(ensemble, cv=3, method="sigmoid")
        final_model.fit(X_train_res, y_train_res)

    # == 5. Saving Model ==
    # Save the final trained model and the imputer for later use in prediction.
    joblib.dump(final_model, config.MODEL_PATH)
    joblib.dump(imputer, config.IMPUTER_PATH)

    # Also save the individual tuned models for later explainability (e.g., SHAP).
    joblib.dump(rf_best, config.RF_PATH)
    joblib.dump(xgb_best, config.XGB_PATH)

    # == 6. Evaluation ==
    # Test the final model's performance on the hold-out test set it has never seen before.
    print("\n--- Final Model Evaluation on Hold-Out Set ---")
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

    return final_model, imputer, X_test, y_test