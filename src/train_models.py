import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from preprocess import load_data, feature_engineering, build_preprocessing_pipeline, split_data

def evaluate_model(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    df = load_data("data/train.csv")
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor, _, _ = build_preprocessing_pipeline(df)

    # Logistic Regression pipeline
    pipe_lr = Pipeline([
        ('preproc', preprocessor),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    pipe_lr.fit(X_train, y_train)
    print("Logistic Regression performance:")
    evaluate_model(pipe_lr, X_test, y_test)

    # Random Forest
    pipe_rf = Pipeline([
        ('preproc', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe_rf.fit(X_train, y_train)
    print("Random Forest performance:")
    evaluate_model(pipe_rf, X_test, y_test)

    # Save the better performing model (example)
    joblib.dump(pipe_rf, "models/pipeline_rf.pkl")
    print("Saved pipeline_rf.pkl")
