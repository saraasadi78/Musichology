"""Shared model loading and prediction utilities for application entry points."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.preprocessing import MENTAL_HEALTH_TARGETS

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "target_models.joblib"


@lru_cache(maxsize=1)
def load_target_models(
    model_path: str | Path = MODEL_PATH,
) -> tuple[dict[str, Any], list[str]]:
    """Load the fitted multi-target pipelines and their raw feature names."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {path}. Run `python -m src.train` first."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError(f"Model bundle at {path} has an unsupported format.")
    if "models" in bundle:
        models = bundle["models"]
    elif set(MENTAL_HEALTH_TARGETS).issubset(bundle):
        models = {
            target: bundle[target]
            for target in MENTAL_HEALTH_TARGETS
        }
    else:
        raise ValueError(f"Model bundle at {path} has an unsupported format.")
    missing = set(MENTAL_HEALTH_TARGETS) - set(models)
    if missing:
        raise ValueError(f"Model bundle is missing targets: {sorted(missing)}")
    features = bundle.get("features")
    if features is None:
        first_model = next(iter(models.values()))
        features = first_model.named_steps["preprocessor"].feature_names_in_.tolist()
    return models, list(features)


def build_feature_frame(
    values: dict[str, Any],
    feature_names: list[str],
) -> pd.DataFrame:
    """Map API/UI values to the exact columns expected by the pipelines."""
    return pd.DataFrame([values]).reindex(columns=feature_names)


def predict_scores(
    values: dict[str, Any],
    model_path: str | Path = MODEL_PATH,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Predict all four mental-health scores from one validated input row."""
    models, feature_names = load_target_models(model_path)
    features = build_feature_frame(values, feature_names)
    scores = {
        target: float(model.predict(features)[0])
        for target, model in models.items()
    }
    return scores, features

