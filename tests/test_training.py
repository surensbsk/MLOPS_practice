import os
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from src.training.train import train_model, save_model_to_s3, S3_BUCKET_NAME, S3_MODEL_KEY


class TestTrainModel:
    def test_returns_random_forest_classifier(self, sample_csv):
        model = train_model(data_path=sample_csv)
        assert isinstance(model, RandomForestClassifier)

    def test_model_is_fitted(self, sample_csv):
        model = train_model(data_path=sample_csv)
        check_is_fitted(model)

    def test_model_has_correct_n_estimators(self, sample_csv):
        model = train_model(data_path=sample_csv)
        assert model.n_estimators == 100

    def test_model_can_predict(self, sample_csv):
        import numpy as np
        model = train_model(data_path=sample_csv)
        dummy_input = pd.DataFrame(
            [np.zeros(len(model.feature_names_in_))],
            columns=model.feature_names_in_,
        )
        preds = model.predict(dummy_input)
        assert preds[0] in (0, 1)

    def test_raises_on_missing_csv(self):
        with pytest.raises(FileNotFoundError):
            train_model(data_path="/nonexistent/path/data.csv")


class TestSaveModelToS3:
    def test_serialises_model_to_disk(self, real_model, tmp_path):
        local_path = str(tmp_path / "model.pkl")
        with patch("src.training.train.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            save_model_to_s3(real_model, local_path=local_path, s3_key="test/model.pkl")
        assert os.path.exists(local_path)

    def test_calls_s3_upload_with_correct_bucket(self, real_model, tmp_path):
        local_path = str(tmp_path / "model.pkl")
        with patch("src.training.train.boto3") as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3
            save_model_to_s3(real_model, local_path=local_path, s3_key=S3_MODEL_KEY)
            mock_s3.upload_file.assert_called_once_with(local_path, S3_BUCKET_NAME, S3_MODEL_KEY)

    def test_calls_s3_upload_exactly_once(self, real_model, tmp_path):
        local_path = str(tmp_path / "model.pkl")
        with patch("src.training.train.boto3") as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3
            save_model_to_s3(real_model, local_path=local_path)
            assert mock_s3.upload_file.call_count == 1

    def test_uses_default_s3_key_when_omitted(self, real_model, tmp_path):
        local_path = str(tmp_path / "model.pkl")
        with patch("src.training.train.boto3") as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.client.return_value = mock_s3
            save_model_to_s3(real_model, local_path=local_path)
            assert mock_s3.upload_file.call_args[0][2] == S3_MODEL_KEY
