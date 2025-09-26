# src/app.py
import streamlit as st
import pandas as pd
import joblib
import os

from preprocessing import encode_features

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

st.title("🚢 Titanic Survival Prediction")

# Load model and scaler
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    st.error("❌ Model not found. Please train the model first using train.py")
    st.stop()

# User input
st.sidebar.header("Enter Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 25)
sibsp = st.sidebar.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.sidebar.number_input("Fare", 0.0, 500.0, 32.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Prepare input data
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked]
})

# Encode categorical
input_data = encode_features(input_data)

# Scale
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)[0]
prob = model.predict_proba(input_scaled)[0][1]

st.subheader("Prediction Result")
if prediction == 1:
    st.success(f"✅ Passenger likely SURVIVED (Probability: {prob:.2f})")
else:
    st.error(f"❌ Passenger likely DID NOT SURVIVE (Probability: {prob:.2f})")
