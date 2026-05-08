# NFL Confidence Pool Agent

An AI-powered assistant for a family NFL confidence pool. Each week it picks the winner of every game and assigns confidence points (1–N) optimally to maximize season-long total score.

## What It Does

- **Predicts** game winners using an XGBoost model trained on 2018–2023 NFL data, with market odds as the primary signal plus adjustments for rest days, injuries, recent form, weather, and strength of schedule
- **Optimizes** confidence point assignments (1–16 regular season, adjusted for playoffs) by sorting games by win probability — highest probability gets highest points
- **Flags** close calls: games within 3% win probability of an adjacent pick are highlighted for manual review
- **Explains** picks via a conversational agent (local Llama 3.3 70B or Claude Sonnet 4.6) that can look up predictions, odds, injuries, and weather in real time
- **Re-ranks** picks when you challenge them, showing a visible diff of what changed and why
- **Refreshes** automatically via cron (Thursday/Friday) on a local VM, with a manual refresh button for breaking news

## Architecture

```
Streamlit dashboard (ui/app.py)
        ↕ HTTP
FastAPI backend (src/api/main.py)
    ├── XGBoost model (src/model/)
    ├── Confidence optimizer (src/optimizer/)
    ├── Conversational agent (src/agent/)
    │     ├── Ollama 70B (default) or Claude Sonnet 4.6
    │     └── 7 tools: schedule, predictions, odds, injuries, weather, assignments, re-rank
    └── SQLite database (data/nfl_pool.db)
```

## Data Sources

| Source | Purpose | Cost |
|--------|---------|------|
| [nfl_data_py](https://github.com/nflverse/nfl_data_py) | Historical game results, schedules, rosters (2018–present) | Free |
| [The Odds API](https://the-odds-api.com) | Market moneylines → implied win probabilities | Free tier (500 req/mo) |
| [Sleeper API](https://docs.sleeper.com) | Weekly injury reports | Free |
| [Open-Meteo](https://open-meteo.com) | Game-day weather for outdoor stadiums | Free |

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [Ollama](https://ollama.com) with `llama3.3:70b` pulled (or a Claude API key for cloud LLM)
- The Odds API key (free at [the-odds-api.com](https://the-odds-api.com))

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

Load historical game data (2018–2025) into SQLite:

```bash
uv run python scripts/ingest_historical.py
```

### Train + Backtest

Train the model on 2018–2023, validate on 2024–2025:

```bash
uv run python scripts/backtest.py
```

Expected output:
```
Training on ~1680 games...
Backtest Results:
  Avg accuracy:      0.58–0.62
  Baseline accuracy: 0.56–0.60
  Avg actual pts/wk: ~95–110
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

### On the VM (always-on)

```bash
uv run bash scripts/start_server.sh
```

Access from your laptop at `http://<vm-local-ip>:8501`.

Add cron jobs (`crontab -e`) to auto-refresh Thursday and Friday:

```cron
@reboot cd /path/to/nfl_agent && bash scripts/start_server.sh
0 9 * * 4 cd /path/to/nfl_agent && uv run python scripts/refresh_weekly.py >> logs/cron.log 2>&1
0 8 * * 5 cd /path/to/nfl_agent && uv run python scripts/refresh_weekly.py >> logs/cron.log 2>&1
```

## Weekly Workflow

1. Cron auto-refreshes Thursday/Friday — dashboard is ready with picks
2. Open the dashboard, review the pick sheet
3. Yellow rows = two picks within 3% win probability — consider swapping
4. Use the chat to challenge any pick ("Why Eagles 16 points?")
5. Agent fetches live data, explains reasoning
6. If you convince it, it re-ranks and shows the diff
7. Submit picks to your pool

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
├── ingest_historical.py   # One-time data backfill
├── backtest.py            # Train + validate model
├── refresh_weekly.py      # Cron entry point
└── start_server.sh        # VM startup script
docs/superpowers/plans/    # Implementation plan
```

## Model Features

The XGBoost model uses 15 features per game:

| Feature | Description |
|---------|-------------|
| `odds_home_win_prob` | Market-implied probability (vig-removed) |
| `home_rest_days` / `away_rest_days` | Days since last game |
| `rest_advantage` | Difference in rest days |
| `home_qb_out` / `away_qb_out` | Starting QB injury flag |
| `home_recent_winpct` / `away_recent_winpct` | Win % last 4 games |
| `home_recent_point_diff` / `away_recent_point_diff` | Avg margin last 4 games |
| `temperature` / `wind_speed` | Game-day weather (outdoor only) |
| `home_sos` / `away_sos` | Strength of schedule (opponent quality) |
| `is_playoff` | Playoff game flag |
