import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Sales Predictor", page_icon="📈")

st.title("📈 Advertising Sales Predictor")
st.write("Enter the advertising spend to predict total sales.")

# Check if model file exists
MODEL_PATH = 'sales_model.pkl'

if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Model file not found!")
    st.info("Please run `python main.py` in your terminal first to train the model and generate the file.")
else:
    # Load the saved model safely
    model = joblib.load(MODEL_PATH)

    # User Inputs
    tv = st.number_input("TV Budget ($000s)", min_value=0.0, value=100.0)
    radio = st.number_input("Radio Budget ($000s)", min_value=0.0, value=20.0)
    newspaper = st.number_input("Newspaper Budget ($000s)", min_value=0.0, value=10.0)

    if st.button("Predict Sales"):
        features = np.array([[tv, radio, newspaper]])
        prediction = model.predict(features)[0]
        st.success(f"The predicted sales are: **{prediction:.2f} thousand units**")

    st.info("This model uses Linear Regression to estimate sales based on historical advertising data.")