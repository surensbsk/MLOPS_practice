import sys
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from sklearn.tree import DecisionTreeClassifier

# Ensure ml-project root is always on sys.path regardless of where pytest is invoked from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def valid_df():
    """10-row DataFrame matching the Social Network Ads schema with correct dtypes."""
    return pd.DataFrame({
        "User ID": pd.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype="int64"),
        "Gender": ["Male", "Female", "Male", "Female", "Male",
                   "Female", "Male", "Female", "Male", "Female"],
        "Age": pd.array([25, 35, 40, 28, 45, 30, 52, 22, 37, 48], dtype="int64"),
        "EstimatedSalary": pd.array(
            [50000, 60000, 80000, 45000, 90000, 55000, 70000, 40000, 75000, 85000],
            dtype="float64",
        ),
        "Purchased": pd.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 1], dtype="int64"),
    })


@pytest.fixture
def mock_model():
    """MagicMock model — use only where joblib.dump is also mocked."""
    model = MagicMock()
    model.predict.return_value = [1]
    return model


@pytest.fixture
def real_model():
    """Minimal fitted DecisionTreeClassifier — safe to joblib.dump."""
    rng = np.random.RandomState(42)
    X = rng.rand(20, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def sample_csv(tmp_path):
    """Synthetic Social Network Ads CSV with proper schema written to a temp file."""
    rng = np.random.RandomState(42)
    n = 40
    df = pd.DataFrame({
        "User ID": range(1, n + 1),
        "Gender": ["Male" if i % 2 == 0 else "Female" for i in range(n)],
        "Age": rng.randint(18, 65, n),
        "EstimatedSalary": rng.randint(15000, 150000, n).astype(float),
        "Purchased": (rng.rand(n) > 0.5).astype(int),
    })
    path = str(tmp_path / "Social_Network_Ads.csv")
    df.to_csv(path, index=False)
    return path
