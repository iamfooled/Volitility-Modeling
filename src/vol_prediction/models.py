"""Model training utilities for volatility forecasting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

TARGET_PREFIX = "target_vol_"


@dataclass
class VolatilityForecastResults:
    """Container holding fitted artifacts and evaluation metrics."""

    model: RegressorMixin
    scaler: StandardScaler
    metrics: Dict[str, float]
    feature_columns: Tuple[str, ...]

    def save(self, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


def _default_model_registry() -> Dict[str, RegressorMixin]:
    return {
        "linear": LinearRegression(),
        "elastic_net": ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=10000),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            random_state=42,
            min_samples_leaf=3,
            n_jobs=-1,
        ),
    }


class VolatilityForecaster:
    """Train scikit-learn regressors on engineered volatility features."""

    def __init__(
        self,
        model_name: str,
        model_registry: Dict[str, RegressorMixin] | None = None,
        n_splits: int = 5,
    ) -> None:
        registry = model_registry or _default_model_registry()
        if model_name not in registry:
            raise KeyError(f"Unknown model: {model_name}. Available: {', '.join(registry)}")

        if n_splits < 2:
            raise ValueError("n_splits must be at least 2 for rolling validation")

        self.model_name = model_name
        self.model = registry[model_name]
        self.n_splits = n_splits

    def fit(self, dataset: pd.DataFrame) -> VolatilityForecastResults:
        target_cols = [c for c in dataset.columns if c.startswith(TARGET_PREFIX)]
        if len(target_cols) != 1:
            raise ValueError(
                "Dataset must contain exactly one target column prefixed with 'target_vol_'"
            )
        target_col = target_cols[0]
        X = dataset.drop(columns=target_cols)
        y = dataset[target_col]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        mae_scores = []
        rmse_scores = []
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model = self._clone_estimator()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae_scores.append(mean_absolute_error(y_test, preds))
            rmse_scores.append(mean_squared_error(y_test, preds, squared=False))

        # Train final model on the full dataset
        self.model.fit(X_scaled, y)

        metrics = {
            "mae": float(np.mean(mae_scores)),
            "rmse": float(np.mean(rmse_scores)),
        }

        return VolatilityForecastResults(
            model=self.model,
            scaler=scaler,
            metrics=metrics,
            feature_columns=tuple(X.columns),
        )

    def predict(self, features: pd.DataFrame, results: VolatilityForecastResults) -> pd.Series:
        missing = set(results.feature_columns) - set(features.columns)
        if missing:
            raise KeyError(f"Missing feature columns: {sorted(missing)}")
        ordered = features[list(results.feature_columns)]
        transformed = results.scaler.transform(ordered)
        predictions = results.model.predict(transformed)
        return pd.Series(predictions, index=features.index)

    def _clone_estimator(self) -> RegressorMixin:
        return joblib.loads(joblib.dumps(self.model))


__all__ = [
    "TARGET_PREFIX",
    "VolatilityForecaster",
    "VolatilityForecastResults",
]
