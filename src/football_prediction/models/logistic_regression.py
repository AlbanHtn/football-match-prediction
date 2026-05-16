"""Logistic Regression match predictor."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseMatchPredictor


class LogisticRegressionPredictor(BaseMatchPredictor):
    """Multinomial Logistic Regression with StandardScaler normalization.

    Wraps the estimator in a Pipeline so that scaling is applied automatically
    during both ``fit`` and ``predict`` — no manual scaling required.
    """

    def __init__(
        self,
        feature_columns: list[str],
        target_column: str = "FTResult",
        **lr_kwargs,
    ) -> None:
        self._lr_kwargs = {
            "multi_class": "multinomial",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42,
            **lr_kwargs,
        }
        super().__init__(feature_columns, target_column)

    def _build_model(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**self._lr_kwargs)),
        ])
