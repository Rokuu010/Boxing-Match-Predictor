# src/style.py
import streamlit as st

def apply_boxing_theme():
    st.markdown("""
        <style>
        /* Make the title big, centered, and red */
        .block-container h1 {
            text-align: center;
            font-size: 3.5rem;
            color: #d32f2f;
            text-transform: uppercase;
            font-weight: 800;
        }
        
        /* Style the 'Predict Winner' button */
        div.stButton > button {
            width: 100%;
            background-color: #d32f2f;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 15px;
            border: 2px solid #b71c1c;
            transition: all 0.3s;
        }
        div.stButton > button:hover {
            background-color: #b71c1c;
            border-color: #ffffff;
            transform: scale(1.02);
        }
        
        /* Style the metric cards to look dark and sleek */
        div[data-testid="stMetric"] {
            background-color: #262730;
            padding: 10px;
            border-radius: 5px;
            border-left: 5px solid #d32f2f;
        }
        </style>
    """, unsafe_allow_html=True)