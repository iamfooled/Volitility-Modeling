# Volatility Prediction Project

This repository hosts a modular pipeline for forecasting market volatility from historical
price data. The project replaces the original prototype notebook with a reproducible and
extensible code base that can be executed from the command line or reused inside other
research workflows.

## Project Goals

* Load historical OHLCV data for an equity index, single stock, or other tradable asset.
* Engineer features that summarize recent price dynamics and realized volatility.
* Train one or more machine learning models to forecast forward-looking volatility.
* Evaluate model performance using configurable rolling-window backtests.

## Project Structure

```
├── README.md                ← Project overview and usage instructions
├── requirements.txt         ← Python dependencies for the volatility pipeline
├── src/                     ← Python package containing reusable components
│   └── vol_prediction/
│       ├── data.py          ← Data loading and feature engineering helpers
│       ├── models.py        ← Forecasting models and evaluation routines
│       └── pipeline.py      ← Command line pipeline for training & evaluation
└── scripts/
    └── run_pipeline.py      ← Thin CLI wrapper for day-to-day experimentation
```

## Getting Started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Prepare data**

   Supply a CSV file that includes the following columns:

   | column        | description                                              |
   |---------------|----------------------------------------------------------|
   | `timestamp`   | Datetime stamp (ISO 8601 or `%Y-%m-%d`)                   |
   | `open`        | Opening price                                            |
   | `high`        | Daily high price                                         |
   | `low`         | Daily low price                                          |
   | `close`       | Closing price                                            |
   | `volume`      | (Optional) traded volume                                 |

   Additional columns are preserved as-is and can be referenced inside custom
   feature engineering functions.

3. **Run the pipeline**

   ```bash
   python scripts/run_pipeline.py \
       --data ./data/spx.csv \
       --target-horizon 5 \
       --model random_forest \
       --output ./artifacts/spx_rf.pkl
   ```

   The command trains the specified model using rolling-window cross validation
   and stores fitted artifacts (scalers, model parameters, evaluation metrics)
   under the `--output` path.

## Extending the Project

* Add new feature generators inside `vol_prediction.data` and register them in
  `FeatureSpec` for automatic inclusion in the pipeline.
* Implement additional forecasting strategies by extending
  `vol_prediction.models.VolatilityForecaster` with new estimators.
* Hook the pipeline into backtesting or live trading infrastructure using the
  `VolatilityForecastResults` data model returned by the CLI.

## License

This project is made available for educational purposes. Adapt the code to fit
your internal research standards and risk-management processes before using it
in production environments.
