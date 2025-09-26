# src/app.py
import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_SEX_PATH = "models/encoder_sex.pkl"
ENCODER_EMBARKED_PATH = "models/encoder_embarked.pkl"

st.title("🚢 Titanic Survival Prediction")

# Load model, scaler, and encoders
if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_SEX_PATH, ENCODER_EMBARKED_PATH]):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le_sex = joblib.load(ENCODER_SEX_PATH)
    le_embarked = joblib.load(ENCODER_EMBARKED_PATH)
else:
    st.error("❌ Model or encoders not found. Please train the model first.")
    st.stop()

st.header("Enter Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 25)
    sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)

with col2:
    parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
    fare = st.number_input("Fare", 0.0, 500.0, 80.0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Predict button
if st.button("🔮 Predict Survival Probability"):
    # Prepare input
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    # Encode categorical with saved LabelEncoders
    input_data['Sex'] = le_sex.transform(input_data['Sex'])
    input_data['Embarked'] = le_embarked.transform(input_data['Embarked'])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Get survival probability
    prob = model.predict_proba(input_scaled)[0][1]  # probability of survival

    # Display probability as percentage
    prob_percent = prob * 100
    st.subheader("Prediction Result")
    st.write(f"⚡ Survival Probability: **{prob_percent:.2f}%**")

    # Color-coded bar
    if prob > 0.7:
        st.success("🟢 High chance of survival")
        st.progress(prob)
    elif prob > 0.4:
        st.warning("🟡 Medium chance of survival")
        st.progress(prob)
    else:
        st.error("🔴 Low chance of survival")
