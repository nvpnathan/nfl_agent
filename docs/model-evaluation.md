# Model Evaluation Guide

For full context on model architecture, training data, and feature definitions, see `docs/model-card.md`.

## Running the Backtest

The backtest retrains per walk-forward fold and evaluates week-by-week on holdout seasons using completed game outcomes.

```bash
uv run python scripts/backtest.py
```

Current folds are defined in `scripts/backtest.py` and use expanding training windows.

## Metrics

Each week produces five metrics, averaged across all weeks in the backtest summary:

| Metric | What it measures |
|---|---|
| `accuracy` | Fraction of games where the model's pick (home if prob ≥ 0.5, away otherwise) was correct |
| `baseline_accuracy` | Fraction of games the home team won in that evaluation set |
| `actual_points` | Points earned that week (confidence points for correct picks, 0 for wrong picks) |
| `expected_points` | Probabilistic points: `Σ win_probability × confidence_points` across all games |
| `brier_score` | Mean squared error of home-win probabilities vs. outcomes (lower is better; 0.25 = coin flip) |

**Interpreting the numbers:**

- `accuracy > baseline_accuracy` means the model adds value over a home-team baseline
- `actual_points / expected_points` close to 1.0 means the model's confidence calibration is realistic
- Lower Brier is better probability quality; 0.25 is coin-flip level

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
- `accuracy` is consistently above `baseline_accuracy`
- `brier_score` is below ~0.25 (coin-flip level)
- `actual_points / expected_points` stays near 1.0 over multiple weeks

**The model needs attention if:**
- `accuracy ≈ baseline_accuracy` (limited lift over baseline)
- `brier_score > 0.25` (probabilities worse than random)
- `actual_points` systematically below `expected_points` (likely overconfidence)
