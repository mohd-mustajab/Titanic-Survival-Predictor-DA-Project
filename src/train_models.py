import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import your preprocessing pipeline builder
from preprocess import build_preprocessing_pipeline

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/pipeline_rf_tuned.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/titanic_synthetic_dirty.csv")

# Function to train and save the model
def train_and_save_model():
    st.warning("Model not found or incompatible. Retraining... ⏳")

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing + Model
    preprocessor, num_feats, cat_feats = build_preprocessing_pipeline(df)
    pipe = preprocessor
    pipe.named_steps["classifier"] = RandomForestClassifier(random_state=42)

    # Train
    pipe.fit(X_train, y_train)

    # Save model
    os.makedirs(os.path.join(os.path.dirname(__file__), "../models"), exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    st.success("Model retrained and saved successfully ✅")
    return pipe


# Try loading model, retrain if error
try:
    pipe = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    pipe = train_and_save_model()


# ---------------- Streamlit UI ----------------
st.title("🚢 Titanic Survival Predictor")

st.write("Enter passenger details below to predict survival:")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.slider("Parents/Children Aboard", 0, 6, 0)
fare = st.slider("Fare", 0, 500, 50)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
title = st.selectbox("Title", ["Mr", "Miss", "Mrs", "Master", "Other"])
cabin = st.selectbox("Cabin", ["A", "B", "C", "D", "E", "F", "G", "T", "Unknown"])

# Collect input
input_data = pd.DataFrame(
    [[pclass, sex, age, sibsp, parch, fare, embarked, title, cabin]],
    columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title", "Cabin"],
)

# Predict
if st.button("Predict Survival"):
    prediction = pipe.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 The passenger is predicted to SURVIVE.")
    else:
        st.error("💀 The passenger is predicted NOT to survive.")