# 🥊 Boxing Match Predictor 🥊
This is a project I built to predict the outcome of professional boxing matches using machine learning. I created an end-to-end pipeline that handles everything from data cleaning and feature engineering to model training and explainability.

To make the model accessible, I built an interactive web application using Streamlit. The app uses a hybrid data system: it relies on a local dataset for core stats but performs live lookups on BoxRec and Wikipedia to get the most up-to-date information for fighters, like their current age and win count.

[Live Link](https://boxing-match-predictor-rokku010.streamlit.app/) 👈



Key Features:

Dynamic Predictions: The app uses an ensemble model (combining XGBoost, Random Forest, and Logistic Regression) that achieves around 87% accuracy on my test set.

Live Data Integration: The system intelligently scrapes BoxRec (primary) and Wikipedia (fallback) to update fighter stats like age and wins, ensuring predictions are based on the most current data available.

Prediction Explanations: I didn't want my model to be simple, so I used the SHAP library to generate feature contribution charts. This explains why the model made a certain prediction, showing which stats were the most influential.

Fallback System: If a fighter can't be found in my local dataset, the app automatically scrapes the web and uses the average stats from my dataset for any missing information, allowing it to make a reasonable prediction for any professional boxer.

Interactive UI: I used Streamlit to build a simple and user-friendly interface that allows anyone to get a prediction without needing to touch the code.

Technology Stack:

Backend & Machine Learning: Python, Pandas, Scikit-learn, XGBoost, SHAP, Imbalanced-learn

Web Application: Streamlit

Web Scraping: Selenium, Beautiful Soup, Requests, Wikipedia

File Management: Git & Git LFS (for handling the large model files)

How to Run This Project Locally:

To get this project running on your own machine, follow these steps.
#

1. Initial Setup

   First, clone the repository to your local machine:

   git clone [https://github.com/Rokuu010/Boxing-Match-Predictor.git](https://github.com/Rokuu010/Boxing-Match-Predictor.git)
cd Boxing-Match-Predictor
#

2. Set Up the Environment

   I used a Python virtual environment to manage the project's dependencies.

# Create the virtual environment
      python -m venv venv

# Activate it (on Windows)
    .\venv\Scripts\activate
#
3. Install Dependencies

   Install all the required libraries from the requirements.txt file.

       pip install -r requirements.txt

(Note: This project uses Selenium, which will automatically download the correct Chrome driver for your browser version).
#

4. Train the Model
   Before you can run the app, you need to train the model. I've created a simple script that handles the entire training pipeline.

       python train.py

This will create all the necessary model files and save them in the models/ directory.
#

5. Run the Web App
   Finally, you can launch the Streamlit web application.

       streamlit run app.py

