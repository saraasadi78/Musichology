"""Regression metrics and SHAP-based model explanations."""

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
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


def _shap_values_array(values: Any) -> np.ndarray:
    """Normalize SHAP's scalar, list, and multi-output return formats."""
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 0]
    return values


def _pipeline_shap_values(
    pipeline: Any,
    features: pd.DataFrame,
    max_background: int = 100,
) -> tuple[Any, np.ndarray, list[str], np.ndarray]:
    """Calculate SHAP values on the transformed features of a sklearn pipeline."""
    import shap

    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    transformed = preprocessor.transform(features)
    feature_names = preprocessor.get_feature_names_out().tolist()
    background = transformed[:max_background]

    try:
        explainer = shap.TreeExplainer(estimator, background)
        explanation = explainer(transformed)
    except Exception:
        explainer = shap.Explainer(
            estimator.predict,
            background,
            feature_names=feature_names,
        )
        explanation = explainer(transformed)
    values = _shap_values_array(explanation.values)
    return explanation, values, feature_names, transformed


def explain_pipeline(
    pipeline: Any,
    features: pd.DataFrame,
    target: str,
    output_dir: str | Path = "results/figures",
    prediction_indices: Sequence[int] = (0,),
    max_background: int = 100,
) -> pd.DataFrame:
    """Save global importance and per-prediction SHAP plots for one target.

    The pipeline must contain ``preprocessor`` and ``model`` steps, as produced
    by :func:`src.train.make_pipeline`. Features are transformed by the fitted
    pipeline before SHAP is calculated, preventing preprocessing leakage.
    """
    import matplotlib.pyplot as plt
    import shap

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    explanation, values, feature_names, transformed = _pipeline_shap_values(
        pipeline,
        features,
        max_background=max_background,
    )
    importance = np.abs(values).mean(axis=0)
    order = np.argsort(importance)[::-1]

    global_table = pd.DataFrame(
        {
            "target": target,
            "feature": np.asarray(feature_names)[order],
            "mean_abs_shap": importance[order],
        }
    )
    figure_height = max(4.0, min(14.0, 0.28 * len(order) + 1.5))
    fig, axis = plt.subplots(figsize=(10, figure_height))
    axis.barh(
        np.asarray(feature_names)[order][::-1],
        importance[order][::-1],
        color="#2a6f97",
    )
    axis.set_title(f"{target}: global SHAP feature importance")
    axis.set_xlabel("Mean absolute SHAP value")
    fig.tight_layout()
    fig.savefig(output / f"{target.lower()}_global_importance.png", dpi=180)
    plt.close(fig)

    for prediction_index in prediction_indices:
        if prediction_index < 0 or prediction_index >= len(features):
            raise IndexError(
                f"Prediction index {prediction_index} is outside 0..{len(features) - 1}."
            )
        waterfall = shap.Explanation(
            values=values[prediction_index],
            base_values=explanation.base_values[prediction_index],
            data=transformed[prediction_index],
            feature_names=feature_names,
        )
        shap.plots.waterfall(waterfall, max_display=15, show=False)
        plt.title(f"{target}: prediction {prediction_index} explanation")
        plt.tight_layout()
        plt.savefig(
            output / f"{target.lower()}_prediction_{prediction_index}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close()

    return global_table


def explain_targets(
    models: Mapping[str, Any],
    features: pd.DataFrame,
    output_dir: str | Path = "results/figures",
    prediction_indices: Sequence[int] = (0,),
    max_background: int = 100,
) -> pd.DataFrame:
    """Generate SHAP plots for every target model and return feature rankings."""
    rankings = [
        explain_pipeline(
            pipeline,
            features,
            target,
            output_dir=output_dir,
            prediction_indices=prediction_indices,
            max_background=max_background,
        )
        for target, pipeline in models.items()
    ]
    return pd.concat(rankings, ignore_index=True)
