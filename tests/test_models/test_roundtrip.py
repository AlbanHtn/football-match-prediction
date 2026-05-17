"""Fit/predict roundtrip tests for all three model implementations."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_prediction.models.logistic_regression import LogisticRegressionPredictor
from football_prediction.models.random_forest import RandomForestPredictor
from football_prediction.models.xgboost_model import XGBoostPredictor

FEATURES = ["f1", "f2", "f3"]
LABELS = ["H", "D", "A"]


def _make_dataset(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.standard_normal((n, len(FEATURES))),
        columns=FEATURES,
    )
    df["FTResult"] = rng.choice(LABELS, size=n)
    return df


@pytest.fixture(scope="module")
def dataset():
    return _make_dataset()


@pytest.mark.parametrize("ModelClass", [
    LogisticRegressionPredictor,
    RandomForestPredictor,
    XGBoostPredictor,
])
class TestModelRoundtrip:
    def test_fit_predict_shape(self, ModelClass, dataset):
        model = ModelClass(feature_columns=FEATURES)
        model.fit(dataset)
        preds = model.predict(dataset)
        assert preds.shape == (len(dataset),)

    def test_predict_labels_are_valid(self, ModelClass, dataset):
        model = ModelClass(feature_columns=FEATURES)
        model.fit(dataset)
        preds = model.predict(dataset)
        assert set(preds).issubset(set(LABELS))

    def test_predict_proba_shape(self, ModelClass, dataset):
        model = ModelClass(feature_columns=FEATURES)
        model.fit(dataset)
        proba = model.predict_proba(dataset)
        assert proba.shape == (len(dataset), 3)

    def test_predict_proba_sums_to_one(self, ModelClass, dataset):
        model = ModelClass(feature_columns=FEATURES)
        model.fit(dataset)
        proba = model.predict_proba(dataset)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_save_load_roundtrip(self, ModelClass, dataset):
        model = ModelClass(feature_columns=FEATURES)
        model.fit(dataset)
        preds_before = model.predict(dataset)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            model.save(path)
            loaded = ModelClass.load(path)

        preds_after = loaded.predict(dataset)
        np.testing.assert_array_equal(preds_before, preds_after)
