from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import boto3
import os

app = FastAPI()

# S3 bucket details
# Full S3 URL: s3://mlopsarp-bk1/mlops_practive/
S3_BUCKET_NAME = "mlopsarp-bk1"
S3_PREFIX = "mlops_practive"
S3_MODEL_PATH = f"{S3_PREFIX}/models/trained_model.pkl"
LOCAL_MODEL_PATH = "trained_model.pkl"

# Function to download the model from S3
def download_model_from_s3():
    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET_NAME, S3_MODEL_PATH, LOCAL_MODEL_PATH)

# Ensure the model is downloaded
if not os.path.exists(LOCAL_MODEL_PATH):
    download_model_from_s3()

# Load the trained model
model = joblib.load(LOCAL_MODEL_PATH)

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.post("/predict")
def predict(request: PredictionRequest):
    input_data = pd.DataFrame([[request.feature1, request.feature2, request.feature3, request.feature4]], 
                               columns=["feature1", "feature2", "feature3", "feature4"])
    prediction = model.predict(input_data)
    return {"prediction": prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)