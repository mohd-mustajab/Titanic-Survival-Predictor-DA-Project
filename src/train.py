# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

from preprocessing import load_data, clean_data, encode_features, scale_features

DATA_PATH = "data/titanic.csv"
MODEL_DIR = "models"

def train_and_evaluate():
    # Load and preprocess
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = encode_features(df)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    # Scale
    X_scaled, scaler = scale_features(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "DecisionTree": DecisionTreeClassifier(random_state=42)
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"=== {name} ===")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        score = model.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_model = model

    # Hyperparameter tuning for DecisionTree
    param_grid = {
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5, 10]
    }
    grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)

    print("Best params (DecisionTree):", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    if grid.best_score_ > best_score:
        best_model = grid.best_estimator_
        best_score = grid.best_score_

    # Save best model and scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"✅ Best model saved with accuracy {best_score:.4f}")

if __name__ == "__main__":
    train_and_evaluate()
