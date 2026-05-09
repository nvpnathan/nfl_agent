# Model Evaluation Guide

## Running the Backtest

The backtest trains on `train_seasons`, then evaluates week-by-week on `val_seasons` using completed game outcomes. Seasons are configured in `config.yaml`.

```bash
uv run python scripts/backtest.py
```

This trains the model and immediately backtests it. To evaluate an already-trained model against different seasons, use the Python API directly (see below).

## Metrics

Each week produces five metrics, averaged across all weeks in the backtest summary:

| Metric | What it measures |
|---|---|
| `accuracy` | Fraction of games where the model's pick (home if prob ≥ 0.5, away otherwise) was correct |
| `baseline_accuracy` | Fraction of games the home team won — the always-pick-home baseline |
| `actual_points` | Points earned that week (confidence points for correct picks, 0 for wrong picks) |
| `expected_points` | Probabilistic points: `Σ win_probability × confidence_points` across all games |
| `brier_score` | Mean squared error of home-win probabilities vs. outcomes (lower is better; 0.25 = coin flip) |

**Interpreting the numbers:**

- `accuracy > baseline_accuracy` means the model adds value over always picking the home team
- `actual_points / expected_points` close to 1.0 means the model's confidence calibration is realistic
- Brier score < 0.23 is roughly what Vegas lines achieve; lower is better

## Python API

To run the backtest programmatically or against custom seasons:

```python
import yaml
from src.model.evaluate import run_season_backtest, compute_week_metrics
import pandas as pd

with open("config.yaml") as f:
    config = yaml.safe_load(f)

results = run_season_backtest(
    model_path=config["model"]["path"],
    seasons=[2024],          # any completed seasons
    db_path=config["db"]["path"],
    point_range=(1, 16),
)

df = pd.DataFrame(results)
print(df[["season", "week", "accuracy", "actual_points", "brier_score"]])
print(df.mean(numeric_only=True))
```

Each element of `results` is a dict with `season`, `week`, and all five metrics above.

## Evaluating a Single Week

```python
from src.model.evaluate import compute_week_metrics

# predictions: list of dicts with home_win_prob, home_win, confidence_points, win_probability
metrics = compute_week_metrics(predictions)
# → {"accuracy": 0.6, "actual_points": 88, "expected_points": 84.2,
#    "brier_score": 0.218, "baseline_accuracy": 0.563}
```

## What to Look For

**The model is working if:**
- `accuracy` is consistently above `baseline_accuracy` (home-team win rate ~57%)
- Brier score is below ~0.25 (coin-flip level)
- `actual_points / expected_points` ratio stays near 1.0 over multiple weeks

**The model needs attention if:**
- `accuracy ≈ baseline_accuracy` — the ML features aren't adding signal over the raw home-field baseline
- Brier score > 0.25 — probabilities are worse than random
- `actual_points` systematically below `expected_points` — model is overconfident

**Known limitation:** Historical training data uses `odds_home_win_prob=0.55` as a default for games without live odds data (most pre-2023 seasons). This weakens the odds feature signal in training. Backtest accuracy improves meaningfully once live odds data is available for the current season.
