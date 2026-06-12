import sys
from unittest.mock import MagicMock, patch

# ── Mock out module-level side effects before api.main is imported ──────────
# api/main.py calls os.path.exists + joblib.load at import time.
# We patch both so no real model file or S3 call is needed.
sys.modules.pop("api.main", None)

_mock_model = MagicMock()
_mock_model.predict.return_value = [1]

with patch("os.path.exists", return_value=True), patch("joblib.load", return_value=_mock_model):
    from api.main import app

from fastapi.testclient import TestClient

client = TestClient(app)

VALID_PAYLOAD = {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0, "feature4": 4.0}


class TestPredictEndpoint:
    def test_valid_input_returns_200(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_response_contains_prediction_key(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert "prediction" in response.json()

    def test_prediction_value_is_numeric(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert isinstance(response.json()["prediction"], (int, float))

    def test_negative_feature_values_accepted(self):
        payload = {"feature1": -1.0, "feature2": -2.0, "feature3": -3.0, "feature4": -4.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_zero_feature_values_accepted(self):
        payload = {"feature1": 0.0, "feature2": 0.0, "feature3": 0.0, "feature4": 0.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_missing_one_feature_returns_422(self):
        payload = {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_empty_payload_returns_422(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_string_feature_value_returns_422(self):
        payload = {**VALID_PAYLOAD, "feature1": "not_a_number"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_extra_fields_are_ignored(self):
        payload = {**VALID_PAYLOAD, "extra_field": "ignored"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_integer_features_coerced_to_float(self):
        payload = {"feature1": 1, "feature2": 2, "feature3": 3, "feature4": 4}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
