"""Abstract base class for all match predictors."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseMatchPredictor(ABC):
    """Common interface for all match outcome prediction models.

    Subclasses must implement ``_build_model`` and may override ``fit`` /
    ``predict`` / ``predict_proba`` for model-specific behaviour.
    """

    def __init__(self, feature_columns: list[str], target_column: str = "FTResult") -> None:
        self.feature_columns = feature_columns
        self.target_column = target_column
        self._model = self._build_model()

    @abstractmethod
    def _build_model(self):
        """Instantiate and return the underlying sklearn-compatible estimator."""
        ...

    def fit(self, df: pd.DataFrame, sample_weight: np.ndarray | None = None) -> "BaseMatchPredictor":
        """Fit the model on a labeled DataFrame.

        Args:
            df: DataFrame containing both features and the target column.
            sample_weight: Optional per-sample weights.

        Returns:
            Self (for method chaining).
        """
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        logger.info("%s: fitting on %d samples", self.__class__.__name__, len(X))
        if sample_weight is not None:
            self._model.fit(X, y, sample_weight=sample_weight)
        else:
            self._model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels."""
        return self._model.predict(df[self.feature_columns].values)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted class probabilities (n_samples × 3)."""
        return self._model.predict_proba(df[self.feature_columns].values)

    def save(self, path: Path) -> None:
        """Serialize the trained model with joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "BaseMatchPredictor":
        """Load a serialized model from disk."""
        model = joblib.load(path)
        logger.info("Model loaded from %s", path)
        return model
