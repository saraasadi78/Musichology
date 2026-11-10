"""Prediction helpers for trained models."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

MENTAL_HEALTH_INDICATORS = ["OCD", "Anxiety", "Depression", "Insomnia"]
NOTEBOOK_GENRE_MAPPING = {
    "Rock": 0,
    "Pop": 1,
    "Hip hop": 2,
    "R&B": 3,
    "Jazz": 4,
    "Country": 5,
    "EDM": 6,
    "K pop": 7,
    "Latin": 8,
    "Lofi": 9,
    "Metal": 10,
    "Classical": 11,
    "Gospel": 12,
    "Folk": 13,
    "Video game music": 14,
}


def predict_linear(
    models: dict[str, Any],
    label_encoder: LabelEncoder,
    fav_genre: str,
    hours_per_day: float,
) -> dict[str, float]:
    """Predict all four scores using the linear-model notebook workflow."""
    genre_code = int(label_encoder.transform([fav_genre])[0])
    features = pd.DataFrame(
        {"Fav genre": [genre_code], "Hours per day": [hours_per_day]}
    )
    return {
        indicator: float(model.predict(features)[0])
        for indicator, model in models.items()
    }


def predict_neural(
    model: Any,
    scaler: StandardScaler,
    age: float,
    fav_genre: str,
    hours_per_day: float,
    genre_mapping: dict[str, int] | None = None,
) -> np.ndarray:
    """Predict four scores using the notebook's manual genre mapping."""
    mapping = genre_mapping or NOTEBOOK_GENRE_MAPPING
    genre_code = mapping.get(fav_genre, -1)
    features = np.array([[age, genre_code, hours_per_day]])
    return np.asarray(model.predict(scaler.transform(features), verbose=0))[0]

