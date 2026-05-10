# Admin Dashboard & Metrics Logging Design

## Overview

Add metrics logging to `train_production.py` and build an admin dashboard (`ui/admin.py`) integrated into the main app's sidebar. The dashboard shows training run history and deep analytics on model vs user picks across all seasons/weeks.

## Files Changed

| File | Change |
|---|---|
| `src/db/schema.py` | Add `model_training_runs` table |
| `src/db/queries.py` | Add 5 new query functions (3 for training runs, 2 for admin analytics) |
| `scripts/train_production.py` | Insert row into `model_training_runs` after training |
| `src/model/evaluate.py` | Add `persist_week_metrics()` function |
| `scripts/backtest.py` | Add `--persist` flag to write metrics to DB |
| `ui/admin.py` | New file — admin dashboard module (2 tabs) |
| `ui/app.py` | Add "Admin" sidebar tab that imports admin module |

## New Database Table: `model_training_runs`

```sql
CREATE TABLE IF NOT EXISTS model_training_runs (
    model_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    cv_accuracy_mean REAL NOT NULL,
    cv_accuracy_std REAL NOT NULL,
    n_samples INTEGER NOT NULL,
    seasons_used TEXT NOT NULL,  -- JSON array, e.g. "[2018,2019,2020,2021,2022,2023]"
    created_at TEXT DEFAULT (datetime('now'))
);
```

`seasons_used` stored as JSON string so the admin can display what data went into each run.

## Task 1: Metrics Logging in `train_production.py`

After `train_model()` returns metrics, insert a row into `model_training_runs`:

```python
from src.db.queries import insert_model_training_run
import json

metrics = train_model(df, model_path)
insert_model_training_run(db_path, {
    "model_version": metrics["model_version"],
    "cv_accuracy_mean": metrics["cv_accuracy_mean"],
    "cv_accuracy_std": metrics["cv_accuracy_std"],
    "n_samples": metrics["n_samples"],
    "seasons_used": json.dumps(seasons),
})
print(f"Training run recorded: {metrics['model_run_id']}")
```

New query function in `src/db/queries.py`:

```python
def insert_model_training_run(db_path: str, run: dict) -> int:
    with _conn(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO model_training_runs
                (model_version, cv_accuracy_mean, cv_accuracy_std, n_samples, seasons_used)
            VALUES (?, ?, ?, ?, ?)
        """, (run["model_version"], run["cv_accuracy_mean"], run["cv_accuracy_std"],
              run["n_samples"], run["seasons_used"]))
        return cursor.lastrowid
```

## Task 2: Persist Backtest Metrics to `model_metrics`

New function in `src/model/evaluate.py`:

```python
def persist_week_metrics(db_path: str, metrics_list: list[dict]) -> int:
    """Insert each week's metrics into model_metrics table. Returns count inserted."""
    with _conn(db_path) as conn:
        for m in metrics_list:
            conn.execute("""
                INSERT INTO model_metrics
                    (season, week, model_version, accuracy, brier_score,
                     expected_points, actual_points, baseline_accuracy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (m["season"], m["week"], MODEL_VERSION,
                  m.get("accuracy"), m.get("brier_score"),
                  m.get("expected_points"), m.get("actual_points"),
                  m.get("baseline_accuracy")))
    return len(metrics_list)
```

`scripts/backtest.py` gets a `--persist` flag that calls this function after computing all metrics.

## Task 3: Admin Dashboard (`ui/admin.py`)

### Structure

- Season selector at top (dropdown, all available seasons)
- Two tabs: **Training Runs** | **Historical Results**

### Tab 1 — Training Runs

- Plotly line chart: CV accuracy mean over time (one point per training run)
- Table below all runs: date, model version, CV accuracy (mean ± std), rows used, seasons

### Tab 2 — Historical Results

#### Season selector (reused from top) + Summary cards
- Weeks covered, overall actual %, model %, override %

#### Week detail view (week selector dropdown)
Three sub-sections:

**Section A — All Picks**

| Game | Pick | Correct? | Model Confidence | Confidence Pts | Tier |
|---|---|---|---|---|---|

Data source: `weekly_submission_picks` + `games.home_win` + `predictions`. Model confidence = `home_win_prob`. Pick source tag: `(model)` or `(override)`.

**Section B — Model vs User Comparison**

| Game | Model Pick | User Pick | Same? | Model Correct? | User Correct? |
|---|---|---|---|---|---|

Games where they diverge are highlighted (amber border).

**Section C — Override Detail**

Only games where `is_overridden=1`:

| Game | Model Pick | User Override To | Model Correct? | User Correct? | Verdict |
|---|---|---|---|---|---|

Verdict: **Override saved you** (model wrong, user right) or **Override hurt you** (model right, user wrong).

Summary count: overrides made, saved, hurt.

**Section D — Season-level override breakdown (aggregate across all weeks)**

| Week | Overrides Made | Saved You | Hurt You |
|---|---|---|---|

Plotly stacked bar chart: saved vs hurt per week.

### Data Sources Summary

| View | Tables Used |
|---|---|
| Training runs chart/table | `model_training_runs` |
| All picks | `weekly_submission_picks` + `games.home_win` + `predictions` |
| Model vs user comparison | `weekly_submission_picks` + `predictions` + `games.home_win` |
| Override detail | `weekly_submission_picks WHERE is_overridden=1` + `predictions` + `games.home_win` |
| Override aggregate by week | `weekly_submission_picks WHERE is_overridden=1` + `predictions` + `games.home_win` |

### Integration with `ui/app.py`

- Admin code lives in `ui/admin.py` as a module
- `ui/app.py` sidebar adds "Admin" tab that imports and renders the admin module
- Admin shares the same season/week state as main app where applicable
