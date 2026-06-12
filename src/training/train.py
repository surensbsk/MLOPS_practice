import os
import sys
import joblib
import boto3
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data_processing.preprocessing import load_and_preprocess_data, split_data

# S3 config — Full URL: s3://mlopsarp-bk1/mlops_practive/
S3_BUCKET_NAME = "mlopsarp-bk1"
S3_PREFIX = "mlops_practive"
S3_MODEL_KEY = f"{S3_PREFIX}/models/trained_model.pkl"
LOCAL_MODEL_PATH = "trained_model.pkl"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "Social_Network_Ads.csv")


def train_model(data_path=DATA_PATH):
    df = load_and_preprocess_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model accuracy: {accuracy:.4f}")
    return model


def save_model_to_s3(model, local_path=LOCAL_MODEL_PATH, s3_key=S3_MODEL_KEY):
    joblib.dump(model, local_path)
    boto3.client("s3").upload_file(local_path, S3_BUCKET_NAME, s3_key)
    print(f"Model uploaded to s3://{S3_BUCKET_NAME}/{s3_key}")


if __name__ == "__main__":
    trained_model = train_model()
    save_model_to_s3(trained_model)
