"""Dataset loading and notebook-compatible preprocessing."""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

MENTAL_HEALTH_TARGETS = ["Anxiety", "Depression", "Insomnia", "OCD"]
LINEAR_FEATURES = ["Fav genre", "Hours per day"]
NEURAL_FEATURES = ["Age", "Fav genre", "Hours per day"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the music and mental-health survey CSV."""
    return pd.read_csv(path)


def prepare_linear_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the filtering and column removal used in ``code.ipynb``."""
    cleaned = data.drop_duplicates().copy()
    cleaned = cleaned.dropna()
    cleaned["Age"] = cleaned["Age"].fillna(cleaned["Age"].mean())
    cleaned = cleaned.drop(
        ["Timestamp", "Permissions", "Foreign languages"],
        axis=1,
    )
    cleaned = cleaned[
        (cleaned["Age"] >= 15)
        & (cleaned["Age"] <= 70)
        & (cleaned["Hours per day"] >= 1)
        & (cleaned["Hours per day"] < 10)
    ].copy()
    cleaned = cleaned[(cleaned["BPM"] < 500) & (cleaned["BPM"] > 20)].copy()
    return cleaned


def prepare_neural_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """Prepare neural-network data exactly as in ``Neural_network.ipynb``."""
    prepared = data.copy()
    label_encoder = LabelEncoder()
    prepared["Fav genre"] = label_encoder.fit_transform(prepared["Fav genre"])
    prepared = prepared.drop(columns=["Timestamp", "Permissions"])
    prepared = prepared.dropna()
    prepared.fillna(prepared.mean(numeric_only=True), inplace=True)
    return prepared, label_encoder

