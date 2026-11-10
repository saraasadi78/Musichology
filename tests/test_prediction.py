"""Tests for prediction helpers."""

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.predict import NOTEBOOK_GENRE_MAPPING, predict_neural


class DummyModel:
    """Minimal model double exposing the Keras prediction interface."""

    def __init__(self) -> None:
        self.received: np.ndarray | None = None

    def predict(self, features: np.ndarray, verbose: int = 0) -> np.ndarray:
        self.received = features
        return np.array([[1.0, 2.0, 3.0, 4.0]])


def test_neural_prediction_scales_features_and_returns_scores() -> None:
    """The predictor uses the notebook mapping and scaler before inference."""
    scaler = StandardScaler().fit(np.array([[20, 0, 1], [40, 14, 5]]))
    model = DummyModel()

    result = predict_neural(model, scaler, 30, "Rock", 4)

    assert result.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert model.received is not None
    assert model.received.shape == (1, 3)
    assert NOTEBOOK_GENRE_MAPPING["Rock"] == 0


def test_unknown_neural_genre_uses_notebook_default() -> None:
    """Unknown genres retain the notebook's -1 fallback."""
    scaler = StandardScaler().fit(np.array([[20, -1, 1], [40, 14, 5]]))
    model = DummyModel()

    predict_neural(model, scaler, 30, "Unknown", 4)

    assert model.received is not None
    assert model.received.shape == (1, 3)
