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


