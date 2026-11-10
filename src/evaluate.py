"""Evaluation metrics for regression models."""

from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def regression_metrics(
    actual: Any,
    predicted: Any,
) -> dict[str, float]:
    """Calculate mean squared error and R-squared."""
    return {
        "mse": float(mean_squared_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def evaluate_linear_models(
    models: dict[str, Any],
    features: Any,
    targets: Any,
) -> dict[str, dict[str, float]]:
    """Evaluate each mental-health regression model."""
    return {
        name: regression_metrics(targets[name], model.predict(features))
        for name, model in models.items()
    }


def evaluate_neural_network(
    model: Any,
    features: Any,
    targets: Any,
) -> dict[str, float]:
    """Evaluate the multi-output network using notebook metrics."""
    predictions = model.predict(features, verbose=0)
    return {
        "mse": float(mean_squared_error(targets, predictions)),
        "r2": float(r2_score(targets, predictions, multioutput="uniform_average")),
    }

