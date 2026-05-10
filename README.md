# NFL Confidence Pool Agent

An AI-powered assistant for a family NFL confidence pool. Each week it picks the winner of every game and assigns confidence points (1–N) optimally to maximize season-long total score.

## What It Does

- **Predicts** weekly NFL winners using a calibrated XGBoost model over market, form, injury, weather, and matchup context features
- **Assigns confidence points** automatically by sorting games on win probability (highest probability gets highest points)
- **Flags uncertainty** for adjacent picks within the configured threshold (`model.uncertainty_threshold`, default 3%)
- **Persists weekly assignments** in SQLite and supports manual point swaps/reverts while preserving a valid permutation
- **Locks weekly submissions** into a snapshot table (`weekly_submissions` + `weekly_submission_picks`) for auditability
- **Refreshes live data** with one command (`scripts/refresh_weekly.py`) or via API (`POST /refresh/{season}/{week}`)

## Architecture

```
Streamlit dashboard (ui/app.py)
        ↕ HTTP
FastAPI backend (src/api/main.py)
    ├── XGBoost model (src/model/)
    ├── Confidence optimizer (src/optimizer/)
    ├── Feature/data pipeline (src/features/, src/data/)
    └── SQLite database (data/nfl_pool.db)
```

## Seasonal Maintenance

Current in-repo maintenance commands:

1. **Weekly refresh:** pull latest schedule/odds/injuries/weather, predict, and persist assignments.
   ```bash
   uv run python scripts/refresh_weekly.py
   ```
2. **Periodic model check:** run walk-forward validation and retraining.
   ```bash
   uv run python scripts/backtest.py
   ```
3. **Deep tuning (optional):** search XGBoost hyperparameters using walk-forward folds.
   ```bash
   uv run python scripts/tune_hyperparams.py
   ```

For exact model training/evaluation workflows, see `docs/model-training.md`, `docs/model-evaluation.md`, and `docs/model-card.md`.

## Data Sources

| Source | Purpose | Cost |
|--------|---------|------|
| [ESPN API](https://site.api.espn.com) | **Primary Source**: Historical results, live scores, box scores, depth charts, and injuries | Free |
| [ESPN Core API](https://sports.core.api.espn.com) | Pre-match betting odds (Bet365 spread/totals) | Free |
| [Open-Meteo](https://open-meteo.com) | Game-day weather for outdoor stadiums | Free |

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [Ollama](https://ollama.com) with `llama3.3:70b` pulled (or a Claude API key for cloud LLM)
- Optional: The Odds API key (legacy helper module in `src/data/odds.py`)

### Install

```bash
uv sync
```

### Configure

Edit `config.yaml`:

```yaml
odds_api:
  key: "YOUR_ODDS_API_KEY"   # get from the-odds-api.com

llm:
  provider: "ollama"         # or "claude"
  ollama_model: "llama3.3:70b"
  claude_model: "claude-sonnet-4-6"
```

Pull the local LLM (if using Ollama):

```bash
ollama pull llama3.3:70b
```

### One-Time Data Ingestion

Populate `data/nfl_pool.db` with completed games, odds, injuries, and team stats before training/backtesting.

This repository currently does not include a dedicated one-shot historical backfill script in `scripts/`.
If you already have a populated DB, place it at `config.yaml -> db.path` (default `data/nfl_pool.db`).

For model details and exact data requirements, see `docs/model-card.md`.

### Train + Backtest

Run walk-forward training + validation:

```bash
uv run python scripts/backtest.py
```

Note: backtest uses temporary fold artifacts by default and does not overwrite `data/model.joblib`.
Use `--write-production-artifact` only if you intentionally want legacy overwrite behavior.

### Train Production Artifact

Train one deployable model artifact (no fold loop):

```bash
uv run python scripts/train_production.py
```

Use configured seasons exactly:

```bash
uv run python scripts/train_production.py --from-config
```

Expected output:
```
Fold: train=[2018, 2019, 2020, 2021] → validate=2022
  Training on <n_games> games...
  2022: accuracy=<...> brier=<...> exp_pts=<...> actual_pts=<...>

Fold: train=[2018, ..., 2022] → validate=2023
  Training on <n_games> games...
  2023: accuracy=<...> brier=<...> exp_pts=<...> actual_pts=<...>

Fold: train=[2018, ..., 2023] → validate=2024
  Training on <n_games> games...
  2024: accuracy=<...> brier=<...> exp_pts=<...> actual_pts=<...>

Fold: train=[2018, ..., 2024] → validate=2025
  Training on <n_games> games...
  2025: accuracy=<...> brier=<...> exp_pts=<...> actual_pts=<...>

=== Walk-forward summary (all folds) ===
  Accuracy:        <...>  (baseline: <...>)
  Brier score:     <...>
  Expected pts/wk: <...>
  Actual pts/wk:   <...>

=== Per-fold summary ===
                 accuracy  brier_score  expected_points  actual_points
fold_val_season
2022                <...>       <...>            <...>          <...>
2023                <...>       <...>            <...>          <...>
2024                <...>       <...>            <...>          <...>
2025                <...>       <...>            <...>          <...>
```

## Running the App

Start the API server and Streamlit dashboard:

```bash
# Terminal 1
uv run uvicorn src.api.main:app --reload --port 8000

# Terminal 2
uv run streamlit run ui/app.py
```

Open `http://localhost:8501` in your browser.

### On a VM (always-on)

Run each service in a process manager (`tmux`, `screen`, `systemd`, etc.) and bind to all interfaces.

```bash
# Terminal 1
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2
uv run streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501
```

Then access Streamlit at `http://<vm-local-ip>:8501`.

Add cron jobs (`crontab -e`) to auto-refresh Thursday and Friday:

```cron
0 9 * * 4 cd /path/to/nfl_agent && uv run python scripts/refresh_weekly.py >> logs/cron.log 2>&1
0 8 * * 5 cd /path/to/nfl_agent && uv run python scripts/refresh_weekly.py >> logs/cron.log 2>&1
```

## Weekly Workflow

1. Run `scripts/refresh_weekly.py` (or let cron run it) to fetch current week data and generate picks
2. Open the dashboard and review assignments ranked by confidence points
3. Check uncertain/close picks and market context (spread + total)
4. Optionally swap confidence points for a game, or revert a game to model ordering
5. Lock the week to snapshot current picks for submission tracking

## Testing

```bash
uv run pytest tests/ -v
```

## File Structure

```
src/
├── db/           # SQLite schema and query functions
├── data/         # API clients (odds, injuries, weather, historical)
├── features/     # Feature engineering pipeline
├── model/        # XGBoost training, prediction, backtesting
├── optimizer/    # Confidence point assignment algorithm
├── agent/        # LLM client, tool suite, conversation loop
└── api/          # FastAPI app
ui/
└── app.py        # Streamlit dashboard
scripts/
├── backtest.py            # Walk-forward training + validation
├── refresh_weekly.py      # Weekly data refresh + prediction generation
├── train_production.py    # Single production artifact training
└── tune_hyperparams.py    # Walk-forward hyperparameter search
docs/superpowers/plans/    # Implementation plan
```

## Model Features

The production model uses 37 features per game (market, rest/form, rolling box-score, injuries, game context, weather, and SOS).

See `docs/model-card.md` for the full canonical feature table, training workflow, evaluation metrics, and limitations.
