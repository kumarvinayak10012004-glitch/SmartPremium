import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import preprocess_predict

# -----------------------------
# Load artifacts
# -----------------------------
model = joblib.load("artifacts/model.pkl")
encoder = joblib.load("artifacts/encoder.pkl")
num_imputer = joblib.load("artifacts/num_imputer.pkl")
cat_imputer = joblib.load("artifacts/cat_imputer.pkl")

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="SmartPremium", layout="centered")
st.title("💰 SmartPremium – Insurance Premium Predictor")
st.write("Enter customer details to predict insurance premium")

# -----------------------------
# Inputs
# -----------------------------
age = st.number_input("Age", 18, 100, 30)
income = st.number_input("Annual Income", 10000, 5000000, 300000)
health = st.slider("Health Score", 0, 100, 70)
claims = st.number_input("Previous Claims", 0, 20, 1)
vehicle_age = st.number_input("Vehicle Age", 0, 30, 5)
credit = st.slider("Credit Score", 300, 900, 650)
duration = st.number_input("Insurance Duration (Years)", 1, 30, 5)

gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.selectbox("Smoking Status", ["Yes", "No"])
exercise = st.selectbox("Exercise Frequency", ["Low", "Medium", "High"])
policy = st.selectbox("Policy Type", ["Basic", "Premium", "Comprehensive"])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Premium 💡"):

    input_df = pd.DataFrame([{
        "Age": age,
        "Annual Income": income,
        "Health Score": health,
        "Previous Claims": claims,
        "Vehicle Age": vehicle_age,
        "Credit Score": credit,
        "Insurance Duration": duration,
        "Gender": gender,
        "Smoking Status": smoking,
        "Exercise Frequency": exercise,
        "Policy Type": policy
    }])

    X = preprocess_predict(
        input_df,
        encoder,
        num_imputer,
        cat_imputer
    )

    prediction = model.predict(X)[0]

    st.success(f"💵 Estimated Insurance Premium: ₹ {prediction:,.2f}")




