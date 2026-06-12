import os
import boto3
import joblib
import pandas as pd
import streamlit as st

# S3 config — Full URL: s3://mlopsarp-bk1/mlops_practive/
S3_BUCKET_NAME = "mlopsarp-bk1"
S3_PREFIX = "mlops_practive"
S3_MODEL_KEY = f"{S3_PREFIX}/models/trained_model.pkl"
LOCAL_MODEL_PATH = "trained_model.pkl"


def download_model_from_s3():
    boto3.client("s3").download_file(S3_BUCKET_NAME, S3_MODEL_KEY, LOCAL_MODEL_PATH)


if not os.path.exists(LOCAL_MODEL_PATH):
    download_model_from_s3()

model = joblib.load(LOCAL_MODEL_PATH)

# Function to predict
def predict(input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit UI
st.title("Social Network Ads Prediction")

# Input fields for user data
age = st.number_input("Age", min_value=18, max_value=100, value=30)
estimated_salary = st.number_input("Estimated Salary", min_value=0, value=50000)
gender = st.selectbox("Gender", options=["Male", "Female"])

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Age': [age],
    'EstimatedSalary': [estimated_salary],
    'Gender': [1 if gender == "Male" else 0]  # Encoding gender
})

# Predict button
if st.button("Predict"):
    prediction = predict(input_data)
    st.write("Prediction:", "Purchased" if prediction[0] == 1 else "Not Purchased")