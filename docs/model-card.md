# NFL Confidence Pool Model Card

## Model Overview

- **Model name:** `xgb_v1`
- **Task type:** Binary classification (`home_win` in `{0,1}`)
- **Prediction target:** Probability that the home team wins an NFL game
- **Primary outputs:**
  - `home_win_prob`
  - `away_win_prob = 1 - home_win_prob`
  - `predicted_winner` (home if `home_win_prob >= 0.5`, else away)
- **Downstream usage:** Predictions are sorted by confidence and mapped to 1..N weekly confidence points in `src/optimizer/confidence.py`

Implementation references:
- `src/model/train.py`
- `src/model/predict.py`
- `src/model/evaluate.py`
- `src/features/builder.py`

## Model Type and Architecture

The production artifact is a calibrated ensemble:

1. **Base learner:** `xgboost.XGBClassifier`
2. **Probability calibration:** `sklearn.calibration.CalibratedClassifierCV(method="sigmoid", cv=5)`

Base hyperparameters (`src/model/train.py`):

```python
XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.02,
    min_child_weight=3,
    subsample=0.88,
    colsample_bytree=0.94,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=2.0,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
```

## Training Data

Training rows are created by `build_training_dataset()` in `src/features/builder.py`.

Inclusion rules:
- Game must be completed (`home_win` is not null)
- Game must have odds in `game_odds` (`home_spread`, `game_total`)

Row-level sources:
- `games` table: schedule, teams, location flags, result label
- `game_odds` table: spread and total
- `team_game_stats`: rolling box-score aggregates
- `injury_reports`: QB availability and key injuries

Label:
- `home_win` (`1` if home team won, `0` otherwise)

Leakage guardrails:
- Feature functions query only prior games (`week < current week`) for rolling stats, form, rest, and SOS in `src/data/historical.py` and `src/db/queries.py`.

## Feature Set (Current)

`FEATURE_COLS` (single source of truth) currently contains **37 features**:

### Market Features

| Feature | Description |
|---|---|
| `home_spread` | Pre-match point spread for home team |
| `game_total` | Pre-match over/under total |

### Rest and Form

| Feature | Description |
|---|---|
| `home_rest_days` | Days since home team's prior game |
| `away_rest_days` | Days since away team's prior game |
| `rest_advantage` | `home_rest_days - away_rest_days` |
| `home_recent_winpct` | Home win% over last 4 completed games |
| `away_recent_winpct` | Away win% over last 4 completed games |
| `home_home_winpct` | Home team's win% in last 4 home games |
| `away_road_winpct` | Away team's win% in last 4 road games |
| `home_recent_point_diff` | Home avg point differential over last 4 games |
| `away_recent_point_diff` | Away avg point differential over last 4 games |

### Box-Score Rolling Features (4-game windows)

| Feature | Description |
|---|---|
| `home_turnover_diff_4wk` | Negated avg turnovers (higher is better) |
| `away_turnover_diff_4wk` | Negated avg turnovers (higher is better) |
| `home_total_yards_4wk` | Avg total offensive yards |
| `away_total_yards_4wk` | Avg total offensive yards |
| `home_third_down_pct_4wk` | Avg third-down conversion rate |
| `away_third_down_pct_4wk` | Avg third-down conversion rate |
| `home_red_zone_pct_4wk` | Avg red-zone conversion rate |
| `away_red_zone_pct_4wk` | Avg red-zone conversion rate |
| `home_pass_yards_4wk` | Avg net passing yards |
| `away_pass_yards_4wk` | Avg net passing yards |
| `home_rush_yards_4wk` | Avg rushing yards |
| `away_rush_yards_4wk` | Avg rushing yards |
| `home_sacks_taken_4wk` | Avg sacks allowed |
| `away_sacks_taken_4wk` | Avg sacks allowed |

### Injury Features

| Feature | Description |
|---|---|
| `home_qb_active` | `0` if QB status is Out/Doubtful/IR/PUP-R, else `1` |
| `away_qb_active` | `0` if QB status is Out/Doubtful/IR/PUP-R, else `1` |
| `home_key_injuries` | Count of out key non-QB players |
| `away_key_injuries` | Count of out key non-QB players |

### Context and Environment

| Feature | Description |
|---|---|
| `is_indoor` | Indoor venue flag |
| `is_neutral` | Neutral-site flag |
| `is_divisional` | Same-division matchup flag |
| `temperature` | Fahrenheit; indoor defaults to `68.0` |
| `wind_speed` | mph; indoor defaults to `0.0` |
| `home_sos` | Opponent-strength proxy (recent opponent win%) |
| `away_sos` | Opponent-strength proxy (recent opponent win%) |
| `is_playoff` | Playoff game flag |

## Missing Data Handling

During training (`src/model/train.py`):
- Features are median-imputed column-wise on the training frame.
- Those medians are stored in the model artifact.

During inference (`src/model/predict.py`):
- Missing feature values are filled from stored medians.
- Remaining missing values are backfilled with `0.5`.

## Model Artifact

Saved via `joblib` at `config["model"]["path"]` (default `data/model.joblib`).

Artifact schema:

```python
{
    "model": <CalibratedClassifierCV>,
    "version": "xgb_v1",
    "features": FEATURE_COLS,
    "medians": {...}
}
```

## Training Workflow

### 1) Walk-forward train + evaluate (recommended)

```bash
uv run python scripts/backtest.py
```

Default behavior uses temporary fold artifacts and does not overwrite `config.model.path`.
To force legacy overwrite behavior:

```bash
uv run python scripts/backtest.py --write-production-artifact
```

What it does:
- Retrains the model in each fold
- Validates on holdout seasons
- Reports weekly and aggregate metrics

Current fold schedule (`scripts/backtest.py`):
- Train 2018-2021 -> validate 2022
- Train 2018-2022 -> validate 2023
- Train 2018-2023 -> validate 2024
- Train 2018-2024 -> validate 2025

### 2) Train a single production artifact

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

Recommended CLI wrapper:

```bash
uv run python scripts/train_production.py
```

Useful flags:

```bash
uv run python scripts/train_production.py --from-config
uv run python scripts/train_production.py --seasons 2018,2019,2020,2021,2022,2023,2024,2025
```

### 3) Optional hyperparameter search

```bash
uv run python scripts/tune_hyperparams.py
```

Notes:
- Uses walk-forward folds in script
- Optimizes Brier score
- Tunes raw XGBoost first (no calibrator during search for speed)

## Evaluation Methodology and Metrics

Weekly metrics are computed in `compute_week_metrics()`:

- `accuracy`: fraction of correct winner picks
- `baseline_accuracy`: home-team win rate for the same set
- `actual_points`: confidence points actually earned from outcomes
- `expected_points`: sum of `win_probability * confidence_points`
- `brier_score`: MSE of predicted home-win probabilities

Backtest logic (`run_season_backtest()`):
- Build historical features for each validation season
- Predict week-by-week
- Assign confidence points by descending win probability
- Join true outcomes and compute metrics

## Testing

Run all tests:

```bash
uv run pytest tests/ -v
```

Current suite size: **62 tests** (`pytest --collect-only`).

Model-relevant coverage:
- `tests/test_model.py`: training artifact creation, probability bounds, spread sensitivity, predict-week output contract, metric calculations
- `tests/test_features.py`: feature list integrity and feature engineering behavior
- `tests/test_optimizer.py`: confidence point ordering, permutation guarantees, uncertainty flags
- `tests/test_odds.py`, `tests/test_espn.py`, `tests/test_db.py`: upstream data parsing and persistence that feed model inputs

## Known Limitations

- Training excludes games without odds rows, so data coverage depends on odds backfill completeness.
- Confidence-point assignment is deterministic sort by win probability (no risk-aware portfolio optimization).
- `get_point_range()` currently returns `(1, n_games)` for all game types.
- There is no dedicated first-party historical ingestion script in `scripts/` at the moment; this branch assumes the SQLite DB is already populated.

## Intended Use and Scope

Intended:
- Weekly winner probabilities and confidence ranking for a season-long NFL confidence pool.

Not intended:
- Sports betting stake sizing
- Real-time in-game win probability
- Player props or fantasy outcomes
