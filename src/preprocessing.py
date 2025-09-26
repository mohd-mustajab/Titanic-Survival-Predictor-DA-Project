# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path: str) -> pd.DataFrame:
    """Load Titanic dataset from CSV file"""
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and preprocess"""
    df = df.copy()

    # Fill missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # Drop Cabin due to too many missing values
    df.drop(columns=['Cabin'], inplace=True, errors='ignore')

    # Drop PassengerId, Name, Ticket (not useful for prediction)
    df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True, errors='ignore')

    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables"""
    df = df.copy()
    le = LabelEncoder()

    for col in ['Sex', 'Embarked']:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    return df

def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    """Standardize numerical features"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
