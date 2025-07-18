import os
import sys
import numpy as np
import streamlit as st

# ✅ Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict_utils import load_model, predict

# Load model
model = load_model()

st.title("🩺 Medical Insurance Cost Estimator")

age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Gender", ["male", "female"])
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.slider("Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}

input_data = [
    age,
    1 if sex == "male" else 0,
    bmi,
    children,
    1 if smoker == "yes" else 0,
    region_map[region]
]

if st.button("Predict Cost"):
    prediction = predict(model, input_data)
    st.success(f"Estimated Medical Cost: ${prediction[0]:,.2f}")
