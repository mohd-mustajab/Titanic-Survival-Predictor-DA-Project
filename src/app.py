import os
import joblib
import pandas as pd
import streamlit as st

# Path to your saved model
MODEL_PATH = os.path.join("models", "pipeline_rf_tuned.pkl")

# Load the model
try:
    pipe = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    pipe = None

st.title("🚢 Titanic Survival Predictor")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert to DataFrame
input_data = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked
}])

# Prediction
if st.button("Predict Survival"):
    if pipe is not None:
        prediction = pipe.predict(input_data)[0]
        prob = pipe.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"✅ Survived (Probability: {prob:.2f})")
        else:
            st.error(f"❌ Did Not Survive (Probability: {prob:.2f})")
    else:
        st.error("Model pipeline is not loaded. Please check MODEL_PATH.")