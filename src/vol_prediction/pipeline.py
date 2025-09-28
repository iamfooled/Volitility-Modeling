"""High level orchestration utilities for the volatility prediction project."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .data import FeatureSpec, load_ohlcv, prepare_training_frame, train_test_split_time_series
from .models import TARGET_PREFIX, VolatilityForecaster


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a volatility forecasting model")
    parser.add_argument("--data", required=True, help="Path to a CSV file with OHLCV data")
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=5,
        help="Number of periods ahead to forecast annualized volatility",
    )
    parser.add_argument(
        "--model",
        choices=["linear", "elastic_net", "random_forest"],
        default="random_forest",
        help="Regression model to train",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of observations reserved for holdout evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./artifacts/vol_forecast.joblib",
        help="File path for the serialized forecast results",
    )
    parser.add_argument(
        "--feature-config",
        type=str,
        default=None,
        help="Optional JSON file specifying custom feature windows",
    )
    return parser


def load_feature_spec(config_path: str | None) -> FeatureSpec:
    if not config_path:
        return FeatureSpec()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(path)

    payload: Dict[str, Any] = json.loads(path.read_text())
    return FeatureSpec(**payload)


def execute_pipeline(
    data_path: str,
    target_horizon: int,
    model_name: str,
    test_size: float,
    output_path: str,
    feature_config: str | None = None,
) -> Dict[str, Any]:
    prices = load_ohlcv(data_path)
    feature_spec = load_feature_spec(feature_config)
    dataset = prepare_training_frame(prices, feature_spec, target_horizon=target_horizon)
    train, test = train_test_split_time_series(dataset, test_size=test_size)

    forecaster = VolatilityForecaster(model_name=model_name)
    results = forecaster.fit(train)
    results.save(output_path)

    target_col = next(c for c in dataset.columns if c.startswith(TARGET_PREFIX))
    predictions = forecaster.predict(test.drop(columns=[target_col]), results)
    evaluation = {
        "holdout_mae": float((test[target_col] - predictions).abs().mean()),
        "holdout_rmse": float(((test[target_col] - predictions) ** 2).mean() ** 0.5),
    }

    summary = {
        "model": model_name,
        "target_horizon": target_horizon,
        "n_observations": len(dataset),
        "metrics": {**results.metrics, **evaluation},
        "artifact_path": str(Path(output_path).resolve()),
        "feature_columns": list(results.feature_columns),
    }
    return summary


def main(argv: list[str] | None = None) -> Dict[str, Any]:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    summary = execute_pipeline(
        data_path=args.data,
        target_horizon=args.target_horizon,
        model_name=args.model,
        test_size=args.test_size,
        output_path=args.output,
        feature_config=args.feature_config,
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
