# Seasonal Model Maintenance Plan (2026)

> **Status:** Historical implementation plan snapshot from 2026-05-10.
> It is preserved for planning history and may reference scripts not present in the current branch.
> Use `README.md` plus `scripts/backtest.py` and `scripts/refresh_weekly.py` for the current operational workflow.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automate the "Result Ingestion -> Feature Refresh -> Walk-Forward Retrain" cycle to ensure the model uses the most recent 4-week rolling stats and adapts to 2026 league trends.

**Architecture:** A new ingestion script closes the data loop by pulling completed box scores into the `team_game_stats` table. The existing training pipeline is then invoked to produce a fresh `model.joblib`. This ensures that Week N+1 predictions are based on Week N reality.

**Tech Stack:** Python 3.11, httpx, SQLite, XGBoost, pandas

---

## Weekly Schedule (Workflow)

| Day | Task | Script |
|---|---|---|
| **Tuesday 9:00 AM** | Ingest Monday Night Football results + Box Scores | `scripts/ingest_results.py` |
| **Tuesday 10:00 AM** | Retrain model (optional but recommended every 2-4 weeks) | `scripts/retrain_model.py` |
| **Wednesday 12:00 PM** | Fetch upcoming odds/injuries and generate picks | `scripts/refresh_weekly.py` |

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `scripts/ingest_results.py` | **Create** | Backfills missing `home_win` and `team_game_stats` for completed games. |
| `scripts/retrain_model.py` | **Create** | Simple wrapper to invoke training on the latest DB state. |
| `docs/superpowers/plans/2026-05-10-seasonal-model-maintenance.md` | **Create** | This plan. |

---

## Task 1: Create Result Ingestion Script

**Files:**
- Create: `scripts/ingest_results.py`

- [ ] **Step 1: Implement `ingest_results.py`**
Use `src.data.espn` to fetch scoreboard data for a specific week and season. If a game is "FINAL", update its score/result in the `games` table and fetch its full `summary` to populate `team_game_stats`.

```python
import yaml
from src.data.espn import fetch_scoreboard, parse_game, fetch_game_summary, parse_box_score
from src.db.queries import insert_espn_game, insert_team_stats

def backfill_completed_games(db_path, season, week):
    events = fetch_scoreboard(season, week)
    print(f"Checking results for {season} Week {week}...")
    
    for event in events:
        game_data = parse_game(event)
        if game_data["home_win"] is not None:
            # Update game result
            insert_espn_game(db_path, game_data)
            
            # Fetch box score for features
            summary = fetch_game_summary(game_data["espn_id"])
            stats = parse_box_score(summary, game_data["espn_id"])
            for s in stats:
                insert_team_stats(db_path, s)
            print(f"  Ingested: {game_data['home_team']} vs {game_data['away_team']}")
```

---

## Task 2: Create Retraining Wrapper

**Files:**
- Create: `scripts/retrain_model.py`

- [ ] **Step 1: Implement `retrain_model.py`**
A script that reads `config.yaml`, builds the training dataset from the current DB, and saves a new model artifact.

```python
import yaml
from src.features.builder import build_training_dataset
from src.model.train import train_model

def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    db_path = config["db"]["path"]
    model_path = config["model"]["path"]
    # Include all completed seasons plus current
    seasons = list(range(2018, 2027)) 
    
    print("Building fresh training dataset...")
    df = build_training_dataset(db_path, seasons)
    
    print(f"Retraining on {len(df)} games...")
    metrics = train_model(df, model_path)
    print(f"Success. CV Accuracy: {metrics['cv_accuracy_mean']:.3f}")
```

---

## Task 3: Create Admin Dashboard

**Files:**
- Create: `ui/admin.py`

- [ ] **Step 1: Implement `ui/admin.py`**
Create a Streamlit dashboard that visualizes model performance over time.
- Query `model_metrics` table to show charts of Accuracy and Brier Score per week.
- Show "Actual vs Expected" points chart to track confidence pool optimization.
- Add a table of "Worst Misses" (games where the model was most confident but wrong).

```python
import streamlit as st
import pandas as pd
import sqlite3
import yaml

def load_metrics(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM model_metrics ORDER BY season, week", conn)
    conn.close()
    return df

def main():
    st.set_page_config(page_title="Model Admin", layout="wide")
    st.title("🏈 Model Performance & Regression Admin")
    
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    db_path = config["db"]["path"]
    
    df = load_metrics(db_path)
    
    # Regression Charts
    st.header("Accuracy & Brier Score")
    st.line_chart(df.set_index('week')[['accuracy', 'baseline_accuracy']])
    st.line_chart(df.set_index('week')['brier_score'])
    
    # Points Tracking
    st.header("Expected vs Actual Points")
    st.bar_chart(df.set_index('week')[['expected_points', 'actual_points']])
```

---

## Task 4: Verification & Dry Run

- [ ] **Step 1: Run Ingestion for 2025 Week 1**
Verify that the `team_game_stats` table populates correctly for a historical week.
- [ ] **Step 2: Check Feature Builder**
Run `build_features_for_game` for a 2025 Week 2 game to ensure it successfully finds the box scores from Week 1 and calculates non-zero averages for `turnover_diff` and `total_yards`.
- [ ] **Step 3: Test Retraining**
Ensure `retrain_model.py` produces a valid `model.joblib` and that `scripts/backtest.py` still runs with the new artifact.
- [ ] **Step 4: Launch Admin UI**
Run `uv run streamlit run ui/admin.py` and verify the metrics charts load correctly from the `model_metrics` table.
