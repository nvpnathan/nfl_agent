# Model Training Guide

This guide documents the current training workflow for the NFL confidence-pool model.

For full model details (architecture, feature list, limitations, evaluation design), see `docs/model-card.md`.

## Overview

The model predicts `P(home team wins)` and is used to rank weekly confidence points.

- Base learner: `XGBClassifier`
- Calibration: `CalibratedClassifierCV(method="sigmoid", cv=5)`
- Feature source of truth: `src/features/builder.py` (`FEATURE_COLS`, currently 37 features)
- Artifact path: `config.yaml -> model.path` (default `data/model.joblib`)

## Prerequisites

```bash
uv sync
mkdir -p data
```

## Train + Walk-Forward Backtest

```bash
uv run python scripts/backtest.py
```

By default, backtest now writes fold model artifacts to a temporary directory so your production artifact is not overwritten.

Legacy behavior (overwrite `config.model.path` on each fold):

```bash
uv run python scripts/backtest.py --write-production-artifact
```

What this script does:
- Trains a fresh model for each walk-forward fold
- Validates week-by-week on the fold holdout season
- Reports accuracy, baseline accuracy, Brier score, expected points, and actual points

Current folds (from `scripts/backtest.py`):
- train `2018-2021` -> validate `2022`
- train `2018-2022` -> validate `2023`
- train `2018-2023` -> validate `2024`
- train `2018-2024` -> validate `2025`

## Train a Single Artifact (Programmatic)

If you want to train one production artifact using configured training seasons:

```bash
uv run python - <<'PY'
import yaml
from src.features.builder import build_training_dataset
from src.model.train import train_model

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

df = build_training_dataset(
    db_path=cfg["db"]["path"],
    seasons=cfg["model"]["train_seasons"],
)
metrics = train_model(df, cfg["model"]["path"])
print(metrics)
PY
```

## Train a Single Artifact (Recommended CLI)

```bash
uv run python scripts/train_production.py
```

Default behavior:
- Auto-discovers seasons that have completed games with odds in SQLite
- Trains one production artifact at `config.model.path`

Optional modes:

```bash
# Use config.model.train_seasons exactly
uv run python scripts/train_production.py --from-config

# Explicit manual override
uv run python scripts/train_production.py --seasons 2018,2019,2020,2021,2022,2023,2024,2025
```

## Optional Hyperparameter Search

```bash
uv run python scripts/tune_hyperparams.py
```

Notes:
- Search optimizes Brier score over walk-forward folds.
- It tunes raw XGBoost parameters first (calibrator skipped for speed during search).

## Data Requirements

Training rows are generated from SQLite and include only games that are:
- Completed (`home_win` present)
- Backed by odds (`game_odds` row exists)

If odds coverage is incomplete, training sample size decreases accordingly.

## Running Tests

```bash
uv run pytest tests/ -v
```

Current test suite size is 62 tests (`pytest --collect-only`).
