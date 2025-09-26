from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import numpy as np

from preprocess import load_data, feature_engineering, build_preprocessing_pipeline, split_data

df = load_data("data/train.csv")
df = feature_engineering(df)
X_train, X_test, y_train, y_test = split_data(df)

preprocessor, _, _ = build_preprocessing_pipeline(df)

# 1) GridSearch for Logistic Regression
pipe_lr = Pipeline([('preproc', preprocessor), ('clf', LogisticRegression(max_iter=2000, solver='liblinear'))])
param_grid_lr = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l1', 'l2']
}
gs_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=5, scoring='f1', n_jobs=-1)
gs_lr.fit(X_train, y_train)
print("Best LR params:", gs_lr.best_params_)
print("LR best CV score:", gs_lr.best_score_)
print("Evaluation on test set:")
print(classification_report(y_test, gs_lr.predict(X_test)))

# 2) RandomizedSearch for RandomForest
pipe_rf = Pipeline([('preproc', preprocessor), ('clf', RandomForestClassifier(random_state=42))])
param_dist = {
    'clf__n_estimators': [50, 100, 200, 400],
    'clf__max_depth': [None, 5, 10, 20],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__max_features': ['sqrt', 0.5, None]
}
rs_rf = RandomizedSearchCV(pipe_rf, param_dist, n_iter=30, cv=5, scoring='f1', random_state=42, n_jobs=-1)
rs_rf.fit(X_train, y_train)
print("Best RF params:", rs_rf.best_params_)
print("RF best CV score:", rs_rf.best_score_)
print("Evaluation on test set:")
print(classification_report(y_test, rs_rf.predict(X_test)))

# Save tuned best model
joblib.dump(rs_rf.best_estimator_, "models/pipeline_rf_tuned.pkl")
print("Saved tuned RF:", "models/pipeline_rf_tuned.pkl")
