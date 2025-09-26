import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

# ================= Feature Engineering =================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Add missing columns with defaults if not present
        if "Name" not in X.columns:
            X["Name"] = "Unknown"
        if "PassengerId" not in X.columns:
            X["PassengerId"] = 0

        # Title feature
        X['Title'] = X['Name'].str.extract(r',\s*([^.]*)\.', expand=False).str.strip()
        X['Title'] = X['Title'].replace(['Mlle','Ms'], 'Miss').replace(['Mme'], 'Mrs').astype(str)
        rare_titles = X['Title'].value_counts()[X['Title'].value_counts() < 10].index
        X['Title'] = X['Title'].replace(rare_titles, 'Rare')

        # Family size and IsAlone
        if 'SibSp' in X.columns and 'Parch' in X.columns:
            X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
            X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
        else:
            X['FamilySize'] = 0
            X['IsAlone'] = 0

        # Drop unused columns
        drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        X = X.drop(columns=[col for col in drop_cols if col in X.columns], errors='ignore')

        return X


# ================= Load Data =================
def load_data(path="data/train.csv"):
    return pd.read_csv(path)


# ================= Build Pipeline =================
def build_pipeline(df):
    df = FeatureEngineer().fit_transform(df)

    numeric_feats = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if 'Survived' in numeric_feats:
        numeric_feats.remove('Survived')

    categorical_feats = df.select_dtypes(include=['object','category','bool']).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats)
    ])

    pipe = Pipeline([
        ("feature_engineer", FeatureEngineer()),  # Important fix
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    return pipe


# ================= Train and Save Model =================
MODEL_PATH = "models/pipeline_rf_tuned.pkl"
DATA_PATH = "data/train.csv"

def train_and_save_model():
    df = load_data(DATA_PATH)
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    pipe = build_pipeline(df)
    pipe.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    return pipe


# ================= Streamlit UI =================
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details below to predict survival:")

# Load or retrain model
try:
    pipe = joblib.load(MODEL_PATH)
except Exception:
    st.warning("Model not found or corrupted. Retraining...")
    pipe = train_and_save_model()

# Input form
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
    if pipe is not None:
        prediction = pipe.predict(input_data)[0]
        if prediction == 1:
            st.success("🎉 The passenger is predicted to SURVIVE.")
        else:
            st.error("💀 The passenger is predicted NOT to survive.")
    else:
        st.error("Model pipeline is not loaded.")
