"""Feature and target construction helpers."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .preprocessing import (
    LINEAR_FEATURES,
    MENTAL_HEALTH_TARGETS,
    NEURAL_FEATURES,
)


def encode_genre(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """Label-encode ``Fav genre`` and return the transformed data and encoder."""
    encoded = data.copy()
    encoder = LabelEncoder()
    encoded["Fav genre"] = encoder.fit_transform(encoded["Fav genre"])
    return encoded, encoder


def make_linear_features(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return notebook-compatible linear-model features and mental-health targets."""
    encoded, _ = encode_genre(data)
    features = encoded[LINEAR_FEATURES]
    targets = encoded[MENTAL_HEALTH_TARGETS]
    return features, targets


def make_neural_features(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return neural-network features and the four regression targets."""
    features = data[NEURAL_FEATURES]
    targets = data[MENTAL_HEALTH_TARGETS]
    return features, targets


def scale_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[StandardScaler, object, object]:
    """Fit the notebook's standard scaler on training features."""
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_features)
    scaled_test = scaler.transform(test_features)
    return scaler, scaled_train, scaled_test

