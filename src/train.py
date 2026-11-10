"""Leakage-safe model benchmarking, tuning, and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocessing import MENTAL_HEALTH_TARGETS, load_data

EstimatorFactory = Callable[[int, dict[str, Any]], Any]


def build_neural_network(input_dim: int = 3) -> Any:
    """Build the Keras architecture used by ``Neural_network.ipynb``."""
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(4, activation="linear"),
        ]
    )


def train_neural_network(
    x_train: Any,
    y_train: pd.DataFrame,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.2,
) -> tuple[Any, Any]:
    """Compile and train the notebook-compatible multi-output network."""
    model = build_neural_network(x_train.shape[1])
    model.compile(optimizer="adam", loss="mean_squared_error")
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=2,
    )
    return model, history


def train_linear_models(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.2,
) -> tuple[
    dict[str, LinearRegression],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Train one linear regression model for each mental-health indicator."""
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=random_state,
    )
    models: dict[str, LinearRegression] = {}
    for target in MENTAL_HEALTH_TARGETS:
        model = LinearRegression()
        model.fit(x_train, y_train[target])
        models[target] = model
    return models, x_train, x_test, y_train, y_test


def make_model_factories() -> dict[str, EstimatorFactory]:
    """Return constructors for all benchmarked model families."""

    def xgboost_factory(random_state: int, params: dict[str, Any]) -> Any:
        from xgboost import XGBRegressor

        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "random_state": random_state,
            "n_jobs": 1,
            "verbosity": 0,
        }
        defaults.update(params)
        return XGBRegressor(**defaults)

    def lightgbm_factory(random_state: int, params: dict[str, Any]) -> Any:
        from lightgbm import LGBMRegressor

        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbosity": -1,
            "random_state": random_state,
            "n_jobs": 1,
        }
        defaults.update(params)
        return LGBMRegressor(**defaults)

    return {
        "Mean": lambda random_state, params: DummyRegressor(
            strategy="mean", **params
        ),
        "Linear": lambda random_state, params: LinearRegression(**params),
        "Ridge": lambda random_state, params: Ridge(
            random_state=random_state, **params
        ),
        "Random Forest": lambda random_state, params: RandomForestRegressor(
            **{
                "n_estimators": 300,
                "random_state": random_state,
                "n_jobs": 1,
                **params,
            }
        ),
        "XGBoost": xgboost_factory,
        "LightGBM": lightgbm_factory,
        # This mirrors the existing network's 64/32 ReLU architecture.
        "Neural Network": lambda random_state, params: MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=100,
            random_state=random_state,
            **params,
        ),
    }


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create fold-fitted numeric imputation and categorical one-hot encoding."""
    numeric = features.select_dtypes(include=["number"]).columns.tolist()
    categorical = features.select_dtypes(exclude=["number"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def make_pipeline(
    features: pd.DataFrame,
    model_name: str,
    random_state: int = 42,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    """Build a model pipeline whose preprocessing is fitted within each fold."""
    factories = make_model_factories()
    if model_name not in factories:
        raise ValueError(f"Unknown model: {model_name}")
    model = factories[model_name](random_state, params or {})
    return Pipeline(
        [
            ("preprocessor", make_preprocessor(features)),
            ("model", model),
        ]
    )


def prepare_benchmark_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Remove targets and identifier fields while retaining missing predictors."""
    excluded = set(MENTAL_HEALTH_TARGETS) | {"Timestamp", "Permissions"}
    features = data.drop(columns=[column for column in excluded if column in data])
    targets = {
        target: data[target].dropna()
        for target in MENTAL_HEALTH_TARGETS
    }
    return features, targets


def _scores(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    cv: KFold,
) -> dict[str, float]:
    """Compute mean five-fold MAE, RMSE, and R²."""
    def root_mean_squared_error(actual: Any, predicted: Any) -> float:
        return float(np.sqrt(mean_squared_error(actual, predicted)))

    scoring = {
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "rmse": make_scorer(root_mean_squared_error, greater_is_better=False),
        "r2": "r2",
    }
    values = cross_validate(
        pipeline,
        features.loc[target.index],
        target,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )
    return {
        "mae": float(-values["test_mae"].mean()),
        "rmse": float(-values["test_rmse"].mean()),
        "r2": float(values["test_r2"].mean()),
    }


def benchmark_models(
    data: pd.DataFrame,
    results_path: str | Path = "results/metrics.csv",
    random_state: int = 42,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Benchmark all requested models and save their cross-validation metrics."""
    features, targets = prepare_benchmark_data(data)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows: list[dict[str, Any]] = []
    for model_name in make_model_factories():
        for target_name, target in targets.items():
            metrics = _scores(
                make_pipeline(features, model_name, random_state),
                features,
                target,
                cv,
            )
            rows.append({"model": model_name, "target": target_name, **metrics})
    result = pd.DataFrame(rows)
    output = Path(results_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def tune_best_model(
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    models_dir: str | Path = "models",
    results_path: str | Path = "results/metrics.csv",
    n_trials: int = 20,
    random_state: int = 42,
) -> tuple[Any, dict[str, Any]]:
    """Tune the best benchmark family with Optuna and save its best pipeline."""
    import optuna

    features, targets = prepare_benchmark_data(data)
    best_row = metrics.loc[metrics["rmse"].idxmin()]
    model_name = str(best_row["model"])
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    def params_for_trial(trial: optuna.Trial) -> dict[str, Any]:
        if model_name == "Ridge":
            return {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}
        if model_name == "Random Forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
        if model_name == "XGBoost":
            return {
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            }
        if model_name == "LightGBM":
            return {
                "num_leaves": trial.suggest_int("num_leaves", 15, 80),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            }
        if model_name == "Neural Network":
            return {
                "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
                "learning_rate_init": trial.suggest_float(
                    "learning_rate_init", 1e-4, 1e-2, log=True
                ),
            }
        return {}

    def objective(trial: optuna.Trial) -> float:
        params = params_for_trial(trial)
        scores = [
            _scores(
                make_pipeline(features, model_name, random_state, params),
                features,
                target,
                cv,
            )["rmse"]
            for target in targets.values()
        ]
        return float(sum(scores) / len(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    tuned_params = study.best_params

    tuned_rows: list[dict[str, Any]] = []
    for target_name, target in targets.items():
        pipeline = make_pipeline(features, model_name, random_state, tuned_params)
        scores = _scores(pipeline, features, target, cv)
        tuned_rows.append(
            {"model": f"Tuned {model_name}", "target": target_name, **scores}
        )

    result = pd.concat([metrics, pd.DataFrame(tuned_rows)], ignore_index=True)
    result.to_csv(results_path, index=False)
    best_tuned = min(tuned_rows, key=lambda row: row["rmse"])
    best_target = str(best_tuned["target"])
    best_pipeline = make_pipeline(features, model_name, random_state, tuned_params)
    best_pipeline.fit(features.loc[targets[best_target].index], targets[best_target])

    artifact = {
        "model": best_pipeline,
        "model_name": model_name,
        "target": best_target,
        "features": features.columns.tolist(),
        "params": tuned_params,
        "study_best_value": study.best_value,
    }
    output_dir = Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_dir / "best_model.joblib")
    return study, artifact


def run_training(
    data_path: str | Path = "data/music-mental.csv",
    results_path: str | Path = "results/metrics.csv",
    models_dir: str | Path = "models",
    n_trials: int = 20,
) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    """Run benchmarking, Optuna tuning, and best-model persistence."""
    data = load_data(data_path)
    metrics = benchmark_models(data, results_path=results_path)
    study, artifact = tune_best_model(
        data,
        metrics,
        models_dir=models_dir,
        results_path=results_path,
        n_trials=n_trials,
    )
    return metrics, study, artifact


if __name__ == "__main__":
    run_training()
