"""
Unit tests for src/feature_selection.py
Run with:  pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `src` importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_selection import (
    evaluate_features,
    get_best_feature,
    load_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Minimal synthetic dataset with two clearly separable crops."""
    rng = np.random.default_rng(0)
    n = 100
    # Crop A has high K; Crop B has low K — K should be the best feature
    records = (
        [{"N": rng.normal(50, 5), "P": rng.normal(50, 5),
          "K": rng.normal(200, 5), "ph": rng.normal(6.5, 0.3), "crop": "CropA"}
         for _ in range(n)]
        + [{"N": rng.normal(50, 5), "P": rng.normal(50, 5),
            "K": rng.normal(20, 5),  "ph": rng.normal(6.5, 0.3), "crop": "CropB"}
           for _ in range(n)]
    )
    return pd.DataFrame(records)


@pytest.fixture()
def split_data(sample_df):
    """Return train/test splits from the sample dataframe."""
    from sklearn.model_selection import train_test_split

    X = sample_df.drop(columns="crop")
    y = sample_df["crop"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ---------------------------------------------------------------------------
# load_data tests
# ---------------------------------------------------------------------------

class TestLoadData:
    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            load_data(tmp_path / "nonexistent.csv")

    def test_raises_on_missing_columns(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        # Missing 'crop' and 'ph' columns
        pd.DataFrame({"N": [1], "P": [2], "K": [3]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            load_data(bad_csv)

    def test_returns_dataframe(self, tmp_path, sample_df):
        csv_path = tmp_path / "soil_measures.csv"
        sample_df.to_csv(csv_path, index=False)
        df = load_data(csv_path)
        assert isinstance(df, pd.DataFrame)
        assert set(["N", "P", "K", "ph", "crop"]).issubset(df.columns)
        assert len(df) == len(sample_df)


# ---------------------------------------------------------------------------
# evaluate_features tests
# ---------------------------------------------------------------------------

class TestEvaluateFeatures:
    def test_returns_all_requested_features(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        features = ["N", "K"]
        result = evaluate_features(X_train, X_test, y_train, y_test, features)
        assert set(result.keys()) == set(features)

    def test_scores_are_between_0_and_1(self, split_data):
        X_train, X_test, y_train, y_test = split_data
        result = evaluate_features(X_train, X_test, y_train, y_test)
        for score in result.values():
            assert 0.0 <= score <= 1.0

    def test_k_is_best_feature_on_separable_data(self, split_data):
        """K should dominate when it perfectly separates the two classes."""
        X_train, X_test, y_train, y_test = split_data
        result = evaluate_features(X_train, X_test, y_train, y_test)
        best, _ = get_best_feature(result)
        assert best == "K", f"Expected 'K' to be best, got '{best}'"


# ---------------------------------------------------------------------------
# get_best_feature tests
# ---------------------------------------------------------------------------

class TestGetBestFeature:
    def test_returns_feature_with_highest_score(self):
        perf = {"N": 0.42, "P": 0.55, "K": 0.91, "ph": 0.33}
        name, score = get_best_feature(perf)
        assert name == "K"
        assert score == 0.91

    def test_handles_single_feature(self):
        perf = {"N": 0.75}
        name, score = get_best_feature(perf)
        assert name == "N"
        assert score == 0.75
