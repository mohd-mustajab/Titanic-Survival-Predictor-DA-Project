import streamlit as st
import joblib
import pandas as pd

st.title("Titanic Survival Predictor")

# Load pipeline
pipe = joblib.load("models/pipeline_rf_tuned.pkl")  # or pipeline_rf.pkl

# Create input widgets
st.sidebar.header("Passenger features")
pclass = st.sidebar.selectbox("Pclass", [1,2,3], index=2)
sex = st.sidebar.selectbox("Sex", ["male","female"])
age = st.sidebar.number_input("Age", value=30, min_value=0, max_value=100)
sibsp = st.sidebar.number_input("SibSp", value=0, min_value=0, max_value=10)
parch = st.sidebar.number_input("Parch", value=0, min_value=0, max_value=10)
fare = st.sidebar.number_input("Fare", value=32.2, min_value=0.0)
embarked = st.sidebar.selectbox("Embarked", ["S","C","Q"])
title = st.sidebar.selectbox("Title", ["Mr","Mrs","Miss","Master","Rare"])
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

input_df = pd.DataFrame([{
    'Pclass': pclass, 'Sex': sex, 'Age': age, 'SibSp': sibsp, 'Parch': parch,
    'Fare': fare, 'Embarked': embarked, 'Title': title, 'FamilySize': family_size, 'IsAlone': is_alone
}])

st.write("Input sample:")
st.write(input_df)

if st.button("Predict survival"):
    pred = pipe.predict(input_df)[0]
    proba = pipe.predict_proba(input_df)[0,1] if hasattr(pipe, "predict_proba") else None
    if pred == 1:
        st.success(f"Predicted: SURVIVED (prob={proba:.2f})" if proba is not None else "Predicted: SURVIVED")
    else:
        st.error(f"Predicted: NOT SURVIVE (prob={proba:.2f})" if proba is not None else "Predicted: NOT SURVIVE")
