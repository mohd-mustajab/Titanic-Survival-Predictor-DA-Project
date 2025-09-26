# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

from preprocessing import load_data, clean_data, scale_features
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/titanic.csv"
MODEL_DIR = "models"

def train_and_evaluate():
    # Load and clean data
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # Encode categorical variables
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Scale numerical features
    X_scaled, scaler = scale_features(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # RandomForest with more trees for stability
    rf = RandomForestClassifier(n_estimators=500, max_depth=None, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print("=== RandomForest Results ===")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model, scaler, and encoders
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(le_sex, os.path.join(MODEL_DIR, "encoder_sex.pkl"))
    joblib.dump(le_embarked, os.path.join(MODEL_DIR, "encoder_embarked.pkl"))

    print("✅ Model, scaler, and encoders saved successfully!")

if __name__ == "__main__":
    train_and_evaluate()
