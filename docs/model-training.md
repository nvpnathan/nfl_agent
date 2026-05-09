# Model Training Guide

## Overview

The model is an XGBoost binary classifier wrapped with isotonic calibration (`CalibratedClassifierCV`). It predicts the probability that the home team wins a given NFL game. Those probabilities feed the confidence point optimizer, which assigns 1–N points per week.

## Prerequisites

```bash
uv sync         # install all dependencies
mkdir -p data   # create the data directory if it doesn't exist
```

## Step 1: Ingest Historical Data (one-time)

Downloads NFL schedule and score data for 2018–2025 from `nfl_data_py` and writes it into SQLite.

```bash
uv run python scripts/ingest_historical.py
```

Expected output: `Loaded ~2200 completed games` then `Ingestion complete.`

This takes 1–3 minutes on first run (network download). Subsequent runs are fast because `nfl_data_py` caches data locally. Running it again is safe — `INSERT OR REPLACE` makes it idempotent.

## Step 2: Train + Backtest

Builds features for training seasons, trains the model, saves it, then backtests on validation seasons.

```bash
uv run python scripts/backtest.py
```

Expected output:

```
Building training dataset for seasons [2018, 2019, 2020, 2021, 2022, 2023]...
Training on ~1650 games...
Training complete: CV accuracy=0.55X

Running backtest on [2024, 2025]...

Backtest Results:
  Avg accuracy:      0.57X
  Baseline accuracy: 0.57X   ← always-home-team baseline
  Avg actual pts/wk: ~70–75
  Avg expected pts:  ~68–72
  Avg Brier score:   ~0.23
```

This takes 3–5 minutes. The trained model is saved to `data/model.joblib`.

## Configuration

Training seasons and the model path are set in `config.yaml`:

```yaml
model:
  path: "data/model.joblib"
  train_seasons: [2018, 2019, 2020, 2021, 2022, 2023]
  val_seasons: [2024, 2025]
  uncertainty_threshold: 0.03
```

To retrain on different seasons, edit `train_seasons` and re-run `scripts/backtest.py`.

## Features

Defined in `src/features/builder.py` as `FEATURE_COLS` — this is the single source of truth imported by training, prediction, and evaluation code.

| Feature | Description |
|---|---|
| `odds_home_win_prob` | Market-implied home win probability (vig-removed) |
| `home_rest_days` | Days since home team's last game |
| `away_rest_days` | Days since away team's last game |
| `rest_advantage` | `home_rest_days - away_rest_days` |
| `home_qb_out` | 1 if home starting QB is Out/Doubtful/IR |
| `away_qb_out` | 1 if away starting QB is Out/Doubtful/IR |
| `home_recent_winpct` | Home team win % over last 4 games |
| `away_recent_winpct` | Away team win % over last 4 games |
| `home_recent_point_diff` | Home team avg point differential over last 4 games |
| `away_recent_point_diff` | Away team avg point differential over last 4 games |
| `temperature` | Game-time temperature in °F (65.0 default for missing/indoor) |
| `wind_speed` | Wind speed in mph (5.0 default for missing/indoor) |
| `home_sos` | Home team strength of schedule (avg opponent win % last 4 games) |
| `away_sos` | Away team strength of schedule |
| `is_playoff` | 1 for wildcard/divisional/conference/Super Bowl |

Historical training data uses `odds_home_win_prob=0.55` as a default when live odds aren't available (pre-2023 seasons).

## Model Architecture

- **Base:** `XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)`
- **Calibration:** `CalibratedClassifierCV(base, cv=5, method="isotonic")`
- **NaN handling:** Missing features filled with per-column training medians (stored in artifact), falling back to 0.5
- **Artifact:** `data/model.joblib` contains `{"model": ..., "version": "xgb_v1", "features": [...], "medians": {...}}`

## Running Tests

```bash
uv run pytest tests/ -v
```

25 tests covering the DB layer, feature engineering, model train/predict, and optimizer.
