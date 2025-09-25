# app.py
import os
import joblib
import streamlit as st
import pandas as pd

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/pipeline_rf_tuned.pkl")

st.title("Titanic Survival Prediction App")

# Step 1 — Check if model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

# Step 2 — Load model
pipe = joblib.load(MODEL_PATH)

# Step 3 — User input
st.sidebar.header("Passenger Data")
Pclass = st.sidebar.selectbox("Pclass", [1, 2, 3])
Sex = st.sidebar.selectbox("Sex", ["male", "female"])
Age = st.sidebar.slider("Age", 0, 80, 30)
SibSp = st.sidebar.slider("SibSp", 0, 5, 0)
Parch = st.sidebar.slider("Parch", 0, 5, 0)
Fare = st.sidebar.number_input("Fare", min_value=0.0, value=32.2)
Embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"])

# Step 4 — Predict button
if st.button("Predict Survival"):
    input_df = pd.DataFrame({
        "Pclass": [Pclass],
        "Sex": [Sex],
        "Age": [Age],
        "SibSp": [SibSp],
        "Parch": [Parch],
        "Fare": [Fare],
        "Embarked": [Embarked]
    })

    prediction = pipe.predict(input_df)[0]
    prob = pipe.predict_proba(input_df)[0][prediction]

    st.write(f"**Prediction:** {'Survived' if prediction == 1 else 'Did not survive'}")
    st.write(f"**Probability:** {prob:.2f}")
