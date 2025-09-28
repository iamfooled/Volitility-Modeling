"""Data loading and feature engineering helpers for volatility forecasting."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class FeatureSpec:
    """Configuration object describing the core feature set.

    Parameters
    ----------
    return_windows:
        Look-back windows (in periods) used to compute cumulative log returns.
    realized_vol_windows:
        Rolling windows used for realized volatility (standard deviation of log returns).
    ema_windows:
        Windows used to compute exponential moving average of volatility estimates.
    additional_generators:
        Callables that accept a price dataframe and return a dataframe of additional
        feature columns aligned with the input index.
    """

    return_windows: Iterable[int] = (1, 5, 10, 21)
    realized_vol_windows: Iterable[int] = (5, 10, 21)
    ema_windows: Iterable[int] = (5, 10)
    additional_generators: List[Callable[[pd.DataFrame], pd.DataFrame]] = field(default_factory=list)

    def build_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return a dataframe of engineered features."""
        if "close" not in prices:
            raise KeyError("Expected a 'close' column in the price dataframe")

        features = pd.DataFrame(index=prices.index)
        log_returns = compute_log_returns(prices["close"])
        features["log_return_1"] = log_returns

        for window in self.return_windows:
            if window <= 1:
                continue
            features[f"cum_return_{window}"] = log_returns.rolling(window).sum()

        for window in self.realized_vol_windows:
            vol = log_returns.rolling(window).std(ddof=0) * np.sqrt(252)
            features[f"realized_vol_{window}"] = vol

        realized_base: Optional[int] = next(iter(self.realized_vol_windows), None)
        if realized_base is not None:
            base_col = f"realized_vol_{realized_base}"
            for window in self.ema_windows:
                features[f"ema_vol_{window}"] = features[base_col].ewm(span=window, adjust=False).mean()

        if "high" in prices and "low" in prices:
            true_range = np.maximum(prices["high"] - prices["low"], 1e-9)
            features["log_range"] = np.log(true_range)

        for generator in self.additional_generators:
            extra = generator(prices)
            if not extra.index.equals(prices.index):
                extra = extra.reindex(prices.index)
            for column in extra.columns:
                if column in features:
                    raise ValueError(f"Duplicate feature column: {column}")
            features = pd.concat([features, extra], axis=1)

        return features


def load_ohlcv(csv_path: str | Path, tz_aware: bool = False) -> pd.DataFrame:
    """Load OHLCV data from a CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise KeyError("CSV must contain a 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=tz_aware)
    df = df.sort_values("timestamp").set_index("timestamp")
    numeric_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["close"])
    return df


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from a price series."""
    return np.log(prices).diff()


def prepare_training_frame(
    prices: pd.DataFrame,
    feature_spec: FeatureSpec,
    target_horizon: int = 5,
    min_history: int = 30,
) -> pd.DataFrame:
    """Combine features and future volatility targets into a single dataframe."""
    if target_horizon < 1:
        raise ValueError("target_horizon must be positive")

    features = feature_spec.build_features(prices)
    log_returns = compute_log_returns(prices["close"])
    future_vol = (
        log_returns.shift(-target_horizon + 1)
        .rolling(target_horizon)
        .std(ddof=0)
        * np.sqrt(252)
    )
    dataset = features.copy()
    dataset[f"target_vol_{target_horizon}"] = future_vol
    dataset = dataset.iloc[min_history:]
    dataset = dataset.dropna()
    return dataset


def train_test_split_time_series(
    dataset: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataset into chronological train/test partitions."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    split_idx = int(len(dataset) * (1 - test_size))
    train = dataset.iloc[:split_idx]
    test = dataset.iloc[split_idx:]
    return train, test


__all__ = [
    "FeatureSpec",
    "compute_log_returns",
    "load_ohlcv",
    "prepare_training_frame",
    "train_test_split_time_series",
]
