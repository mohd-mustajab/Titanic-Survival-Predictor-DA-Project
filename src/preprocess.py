import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

def load_data(path="data/train.csv"):
    df = pd.read_csv(path)
    return df
def feature_engineering(df):
    # Example feature engineering for Titanic
    df = df.copy()
    # Title from Name
    df['Title'] = df['Name'].str.extract(r',\s*([^.]*)\.', expand=False).str.strip()
    # Simplify titles
    df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss').replace(['Mme'], 'Mrs').astype(str)
    rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    # Drop columns not used
    drop_cols = ['PassengerId','Name','Ticket','Cabin']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df

def build_preprocessing_pipeline(df):
    # Identify numeric and categorical columns
    numeric_feats = df.select_dtypes(include=['int64','float64']).columns.tolist()
    # remove target if present
    if 'Survived' in numeric_feats:
        numeric_feats.remove('Survived')
    categorical_feats = df.select_dtypes(include=['object','category','bool']).columns.tolist()

    # Numerical transformer: impute median + scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical: impute most frequent + one-hot
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats)
    ])
    return preprocessor, numeric_feats, categorical_feats

def split_data(df, target='Survived', test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

if __name__ == "__main__":
    df = load_data()
    df = feature_engineering(df)
    preprocessor, num_feats, cat_feats = build_preprocessing_pipeline(df)
    X_train, X_test, y_train, y_test = split_data(df)
    # Save an example pipeline (without a model) if you want
    joblib.dump(preprocessor, "models/preprocessor_example.pkl")
    print("Saved preprocessor, numeric:", num_feats, "categorical:", cat_feats)