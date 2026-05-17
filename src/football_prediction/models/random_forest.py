"""Random Forest match predictor."""

from sklearn.ensemble import RandomForestClassifier

from .base import BaseMatchPredictor


class RandomForestPredictor(BaseMatchPredictor):
    """Random Forest classifier for 3-class match outcome prediction.

    Default hyperparameters are tuned for the Big5 dataset (~32k samples,
    ~36 features). Adjust via ``rf_kwargs``.
    """

    def __init__(
        self,
        feature_columns: list[str],
        target_column: str = "FTResult",
        **rf_kwargs,
    ) -> None:
        self._rf_kwargs = {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "random_state": 42,
            "n_jobs": -1,
            **rf_kwargs,
        }
        super().__init__(feature_columns, target_column)

    def _build_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(**self._rf_kwargs)
