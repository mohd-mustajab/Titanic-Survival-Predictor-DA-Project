# src/train.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from clean_data import clean_titanic_data

DATA_PATH = "data/titanic.csv"
MODEL_DIR = "models"

# Step 1: Clean data
df = clean_titanic_data(DATA_PATH)

# Step 2: Encode categorical variables
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

# Step 3: Separate features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Step 4: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Train RandomForest
model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Step 8: Save model, scaler, and encoders
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, "titanic_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(le_sex, os.path.join(MODEL_DIR, "encoder_sex.pkl"))
joblib.dump(le_embarked, os.path.join(MODEL_DIR, "encoder_embarked.pkl"))

print("✅ Model, scaler, and encoders saved successfully!")
