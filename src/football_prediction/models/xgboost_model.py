"""XGBoost match predictor."""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .base import BaseMatchPredictor

# XGBoost requires integer class labels — map FTResult strings to ints
_LABEL_MAP = {"H": 0, "D": 1, "A": 2}
_LABEL_INV = {v: k for k, v in _LABEL_MAP.items()}


class XGBoostPredictor(BaseMatchPredictor):
    """XGBoost classifier for 3-class match outcome prediction.

    XGBoost requires numeric class labels. This class handles the
    H/D/A ↔ 0/1/2 mapping transparently so the API remains identical
    to other predictors.
    """

    def __init__(
        self,
        feature_columns: list[str],
        target_column: str = "FTResult",
        **xgb_kwargs,
    ) -> None:
        self._xgb_kwargs = {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "mlogloss",
            **xgb_kwargs,
        }
        super().__init__(feature_columns, target_column)

    def _build_model(self) -> XGBClassifier:
        return XGBClassifier(**self._xgb_kwargs)

    def fit(self, df: pd.DataFrame, sample_weight: np.ndarray | None = None) -> "XGBoostPredictor":
        X = df[self.feature_columns].values
        y = df[self.target_column].map(_LABEL_MAP).values
        if sample_weight is not None:
            self._model.fit(X, y, sample_weight=sample_weight)
        else:
            self._model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        numeric_preds = self._model.predict(df[self.feature_columns].values)
        return np.array([_LABEL_INV[p] for p in numeric_preds])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(df[self.feature_columns].values)
