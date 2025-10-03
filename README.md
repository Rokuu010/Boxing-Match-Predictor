# 🥊 Boxing Match Predictor 🥊

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

This project is an end-to-end machine learning application designed to predict the outcomes of professional boxing matches. It features a complete data pipeline, from data collection and feature engineering to model training, evaluation, and explainability.

To make the model accessible, it's deployed as an interactive web application using Streamlit. The app uses a hybrid data system, relying on a local dataset for core stats while performing live lookups on BoxRec and Wikipedia to ensure predictions are based on the most current fighter information available.


*(A screenshot or GIF of your app here would look great!)*

---

##  Live App

**Try the interactive predictor yourself:** [**boxing-match-predictor.streamlit.app**](https://boxing-match-predictor-rokku010.streamlit.app/)

---

##  Key Features

*  **Ensemble Model Predictions:** Utilises a powerful ensemble of **XGBoost, Random Forest, and Logistic Regression** models to achieve ~87% prediction accuracy on the test set.

*  **Live Data Integration:** Fetches up-to-date fighter stats (age, wins, losses) in real-time by scraping **BoxRec** and **Wikipedia**, ensuring predictions are always current.

*  **Explainable AI:** Generates **SHAP** feature contribution charts to explain *why* a prediction was made, providing transparency and insight into the model's decision-making process.

*  **Fallback System:** If a fighter isn't in the local dataset, the app automatically scrapes their data and imputes any missing stats using dataset averages, allowing it to make a reasonable prediction for almost any professional boxer.

*  **Interactive UI:** A clean and user-friendly interface built with **Streamlit** that allows anyone to easily input two fighters and get an instant prediction.

---

## 🛠️ Technology Stack

* **Data Science & Machine Learning:**
    `Python`, `Pandas`, `Scikit-learn`, `XGBoost`, `SHAP`, `Imbalanced-learn`

* **Web Application & Scraping:**
    `Streamlit`, `Selenium`, `Beautiful Soup`, `Requests`, `Wikipedia`

* **Version Control:**
    `Git`, `Git LFS` (for handling large model files)

---

##  Setup:

Follow these steps to set up and run the project on your local machine.

1.  **Prerequisites**

    Ensure you have **Python 3.9** or later and **Git** installed on your system.

2.  **Clone the Repository**

    Open your terminal, navigate to your desired directory, and clone the repository.

    ```sh
    git clone [https://github.com/Rokuu010/Boxing-Match-Predictor.git]
    
    cd Boxing-Match-Predictor
    ```

3.  **Set Up a Virtual Environment**

    It's highly recommended to use a virtual environment to manage project dependencies.

    ```sh
    # Create the virtual environment
    python -m venv venv

    # Activate the environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

4.  **Install Dependencies**

    Install all the required libraries from the `requirements.txt` file.

    ```sh
    pip install -r requirements.txt
    ```
    *(Note: Selenium will automatically download the correct Chrome driver for your browser.)*

5.  **Train the Model**

    Run the training script. This will process the data and generate the machine learning models.

    ```sh
    python train.py
    ```
    This will create the necessary model files and save them in the `models/` directory.

6.  **Run the Web App**

    Finally, launch the Streamlit application.

    ```sh
    streamlit run app.py
    ```
    Your browser should automatically open with the application running locally.
