import pytest
import pandas as pd
from src.data_processing.data_validation import (
    validate_data,
    check_data_integrity,
    validate_and_process_data,
)


class TestValidateData:
    def test_valid_df_returns_true(self, valid_df):
        assert validate_data(valid_df) is True

    def test_raises_on_missing_values(self, valid_df):
        valid_df.loc[0, "Age"] = None
        with pytest.raises(ValueError, match="missing values"):
            validate_data(valid_df)

    def test_raises_type_error_on_wrong_age_dtype(self, valid_df):
        valid_df["Age"] = valid_df["Age"].astype(str)
        with pytest.raises(TypeError, match="Age"):
            validate_data(valid_df)

    def test_raises_type_error_on_wrong_salary_dtype(self, valid_df):
        valid_df["EstimatedSalary"] = valid_df["EstimatedSalary"].astype(str)
        with pytest.raises(TypeError, match="EstimatedSalary"):
            validate_data(valid_df)

    def test_raises_type_error_on_wrong_purchased_dtype(self, valid_df):
        valid_df["Purchased"] = valid_df["Purchased"].astype(float)
        with pytest.raises(TypeError, match="Purchased"):
            validate_data(valid_df)

    def test_raises_on_missing_column(self, valid_df):
        valid_df = valid_df.drop(columns=["Age"])
        with pytest.raises(KeyError):
            validate_data(valid_df)


class TestCheckDataIntegrity:
    def test_valid_purchased_returns_true(self, valid_df):
        assert check_data_integrity(valid_df) is True

    def test_all_zeros_purchased_passes(self, valid_df):
        valid_df["Purchased"] = 0
        assert check_data_integrity(valid_df) is True

    def test_all_ones_purchased_passes(self, valid_df):
        valid_df["Purchased"] = 1
        assert check_data_integrity(valid_df) is True

    def test_raises_on_value_two(self, valid_df):
        valid_df.loc[0, "Purchased"] = 2
        with pytest.raises(ValueError, match="Invalid values"):
            check_data_integrity(valid_df)

    def test_raises_on_negative_value(self, valid_df):
        valid_df.loc[0, "Purchased"] = -1
        with pytest.raises(ValueError, match="Invalid values"):
            check_data_integrity(valid_df)


class TestValidateAndProcessData:
    def test_returns_same_df_for_valid_input(self, valid_df):
        result = validate_and_process_data(valid_df)
        assert result.equals(valid_df)

    def test_raises_for_missing_values(self, valid_df):
        valid_df.loc[0, "Age"] = None
        with pytest.raises(ValueError):
            validate_and_process_data(valid_df)

    def test_raises_for_invalid_purchased(self, valid_df):
        valid_df.loc[0, "Purchased"] = 5
        with pytest.raises(ValueError):
            validate_and_process_data(valid_df)
