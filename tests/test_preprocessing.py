import pytest
import pandas as pd
from src.data_processing.preprocessing import (
    preprocess_data,
    split_data,
    load_and_preprocess_data,
)


class TestPreprocessData:
    def test_drops_nan_rows(self, valid_df):
        valid_df.loc[0, "Age"] = None
        result = preprocess_data(valid_df.copy())
        assert result.isnull().values.any() == False
        assert len(result) == len(valid_df) - 1

    def test_encodes_gender_drops_original_column(self, valid_df):
        result = preprocess_data(valid_df.copy())
        assert "Gender" not in result.columns

    def test_gender_encoding_creates_dummy_column(self, valid_df):
        result = preprocess_data(valid_df.copy())
        gender_cols = [c for c in result.columns if c.startswith("Gender_")]
        assert len(gender_cols) == 1

    def test_age_is_scaled(self, valid_df):
        result = preprocess_data(valid_df.copy())
        assert abs(result["Age"].mean()) < 1e-9

    def test_salary_is_scaled(self, valid_df):
        result = preprocess_data(valid_df.copy())
        assert abs(result["EstimatedSalary"].mean()) < 1e-9

    def test_purchased_column_preserved(self, valid_df):
        result = preprocess_data(valid_df.copy())
        assert "Purchased" in result.columns

    def test_returns_dataframe(self, valid_df):
        result = preprocess_data(valid_df.copy())
        assert isinstance(result, pd.DataFrame)


class TestSplitData:
    def test_row_count_is_preserved(self, valid_df):
        df = preprocess_data(valid_df.copy())
        X_train, X_test, y_train, y_test = split_data(df)
        assert len(X_train) + len(X_test) == len(df)

    def test_purchased_not_in_features(self, valid_df):
        df = preprocess_data(valid_df.copy())
        X_train, X_test, y_train, y_test = split_data(df)
        assert "Purchased" not in X_train.columns
        assert "Purchased" not in X_test.columns

    def test_target_series_values_are_binary(self, valid_df):
        df = preprocess_data(valid_df.copy())
        _, _, y_train, y_test = split_data(df)
        all_values = pd.concat([y_train, y_test])
        assert set(all_values.unique()).issubset({0, 1})

    def test_default_test_size_is_20_percent(self, valid_df):
        df = preprocess_data(valid_df.copy())
        X_train, X_test, _, _ = split_data(df)
        assert len(X_test) == pytest.approx(len(df) * 0.2, abs=1)

    def test_custom_test_size(self, valid_df):
        df = preprocess_data(valid_df.copy())
        X_train, X_test, _, _ = split_data(df, test_size=0.3)
        assert len(X_test) == pytest.approx(len(df) * 0.3, abs=1)


class TestLoadAndPreprocessData:
    def test_returns_dataframe(self, sample_csv):
        result = load_and_preprocess_data(sample_csv)
        assert isinstance(result, pd.DataFrame)

    def test_no_missing_values(self, sample_csv):
        result = load_and_preprocess_data(sample_csv)
        assert result.isnull().values.any() == False

    def test_gender_column_encoded(self, sample_csv):
        result = load_and_preprocess_data(sample_csv)
        assert "Gender" not in result.columns

    def test_row_count_is_positive(self, sample_csv):
        result = load_and_preprocess_data(sample_csv)
        assert len(result) > 0
