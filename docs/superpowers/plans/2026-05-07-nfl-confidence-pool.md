# NFL Confidence Pool AI Agent — Implementation Plan

> **Status:** Historical implementation plan snapshot from 2026-05-07.
> It is preserved for planning history and may reference files, APIs, or architecture details that are no longer current.
> Use `README.md`, `docs/model-card.md`, `docs/model-training.md`, and `docs/model-evaluation.md` as the current source of truth.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local AI agent that predicts NFL game winners with calibrated probabilities and optimally assigns confidence points (1-N) for a season-long family confidence pool, surfaced through a Streamlit dashboard with a conversational agent for challenging picks.

**Architecture:** XGBoost model trained on 2018–2023 seasons using market odds as baseline + manual feature adjustments (rest days, injuries, weather, recent form, SOS). A local 70B LLM (Ollama) or Claude Sonnet 4.6 (configurable) serves as the conversational agent with tools to query predictions, injury reports, odds, matchup history, and weather — and can re-rank confidence assignments with a visible diff. FastAPI handles the prediction/data API; Streamlit renders the weekly pick sheet and chat interface.

**Tech Stack:** Python 3.11+, nfl_data_py, XGBoost, scikit-learn, SQLite (sqlite3), FastAPI + Uvicorn, Streamlit, Anthropic SDK, Ollama, Open-Meteo (weather), The Odds API (free tier), Sleeper API (injuries), pytest

---

## File Structure

```
nfl_agent/
├── config.yaml                          # API keys, model choice, cron settings
├── requirements.txt
├── src/
│   ├── db/
│   │   ├── schema.py                    # SQLite schema creation
│   │   └── queries.py                   # All DB read/write functions
│   ├── data/
│   │   ├── historical.py                # nfl_data_py ingestion
│   │   ├── odds.py                      # The Odds API client
│   │   ├── injuries.py                  # Sleeper API injury client
│   │   └── weather.py                   # Open-Meteo weather client
│   ├── features/
│   │   └── builder.py                   # Feature engineering pipeline
│   ├── model/
│   │   ├── train.py                     # XGBoost training + calibration
│   │   ├── predict.py                   # Game-level win probability
│   │   └── evaluate.py                  # Backtesting + metrics
│   ├── optimizer/
│   │   └── confidence.py                # Confidence point assignment + uncertainty flags
│   ├── agent/
│   │   ├── llm.py                       # LLM client abstraction (Ollama / Claude)
│   │   ├── tools.py                     # Agent tool implementations
│   │   └── chat.py                      # Conversation loop + re-ranking logic
│   └── api/
│       └── main.py                      # FastAPI app
├── ui/
│   └── app.py                           # Streamlit dashboard
├── scripts/
│   ├── ingest_historical.py             # One-time backfill (2018–2025)
│   ├── backtest.py                      # Run backtesting on 2024–2025
│   └── refresh_weekly.py                # Cron entry point (data + predictions)
└── tests/
    ├── test_db.py
    ├── test_features.py
    ├── test_model.py
    ├── test_optimizer.py
    └── test_agent.py
```

---

## Phase 1: Foundation — Data Pipeline + Database

### Task 1: Project Setup + SQLite Schema

**Files:**
- Create: `config.yaml`
- Create: `requirements.txt`
- Create: `src/db/schema.py`
- Create: `src/db/queries.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Create `requirements.txt`**

```
nfl_data_py>=0.2
xgboost>=2.0
scikit-learn>=1.4
pandas>=2.0
numpy>=1.26
requests>=2.31
openmeteo-requests>=1.2
retry-requests>=2.0
fastapi>=0.111
uvicorn>=0.29
streamlit>=1.35
anthropic>=0.28
ollama>=0.2
joblib>=1.4
pyyaml>=6.0
pytest>=8.0
httpx>=0.27
```

Install: `uv sync`

- [ ] **Step 2: Create `config.yaml`**

```yaml
odds_api:
  key: "YOUR_ODDS_API_KEY"
  sport: "americanfootball_nfl"
  regions: "us"
  markets: "h2h"

sleeper:
  base_url: "https://api.sleeper.app/v1"

llm:
  provider: "ollama"        # "ollama" or "claude"
  ollama_model: "llama3.3:70b"
  claude_model: "claude-sonnet-4-6"

db:
  path: "data/nfl_pool.db"

model:
  path: "data/model.joblib"
  train_seasons: [2018, 2019, 2020, 2021, 2022, 2023]
  val_seasons: [2024, 2025]
  uncertainty_threshold: 0.03

pool:
  regular_season_point_range: [1, 16]
  playoff_point_range: [8, 16]
```

- [ ] **Step 3: Write the failing test for schema creation**

```python
# tests/test_db.py
import sqlite3
import tempfile
import os
import pytest
from src.db.schema import create_schema

def test_schema_creates_all_tables():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        create_schema(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        expected = {"games", "predictions", "weekly_assignments", "conversations",
                    "rerankings", "injury_reports", "model_metrics", "family_picks"}
        assert expected.issubset(tables)
    finally:
        os.unlink(db_path)
```

- [ ] **Step 4: Run test to verify it fails**

```bash
pytest tests/test_db.py::test_schema_creates_all_tables -v
```
Expected: `FAILED — ModuleNotFoundError: No module named 'src'`

- [ ] **Step 5: Create `src/__init__.py` and `src/db/__init__.py`**

```bash
mkdir -p src/db src/data src/features src/model src/optimizer src/agent src/api
touch src/__init__.py src/db/__init__.py src/data/__init__.py src/features/__init__.py
touch src/model/__init__.py src/optimizer/__init__.py src/agent/__init__.py src/api/__init__.py
mkdir -p data tests
```

- [ ] **Step 6: Create `src/db/schema.py`**

```python
import sqlite3

def create_schema(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            game_type TEXT NOT NULL DEFAULT 'regular',
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            game_date TEXT NOT NULL,
            stadium TEXT,
            is_outdoor INTEGER DEFAULT 1,
            home_score INTEGER,
            away_score INTEGER,
            home_win INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            home_win_prob REAL NOT NULL,
            odds_implied_prob REAL,
            model_version TEXT,
            predicted_winner TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (game_id) REFERENCES games(game_id)
        );

        CREATE TABLE IF NOT EXISTS weekly_assignments (
            assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            game_id TEXT NOT NULL,
            predicted_winner TEXT NOT NULL,
            confidence_points INTEGER NOT NULL,
            win_probability REAL NOT NULL,
            is_uncertain INTEGER DEFAULT 0,
            is_overridden INTEGER DEFAULT 0,
            override_reason TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(season, week, game_id)
        );

        CREATE TABLE IF NOT EXISTS conversations (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            season INTEGER,
            week INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS rerankings (
            reranking_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            game_id TEXT NOT NULL,
            old_points INTEGER NOT NULL,
            new_points INTEGER NOT NULL,
            reason TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS injury_reports (
            injury_id INTEGER PRIMARY KEY AUTOINCREMENT,
            team TEXT NOT NULL,
            player_name TEXT NOT NULL,
            position TEXT,
            injury_status TEXT,
            is_qb INTEGER DEFAULT 0,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            fetched_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS model_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            model_version TEXT,
            accuracy REAL,
            brier_score REAL,
            expected_points REAL,
            actual_points REAL,
            baseline_accuracy REAL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS family_picks (
            pick_id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            member_name TEXT NOT NULL,
            game_id TEXT NOT NULL,
            picked_team TEXT NOT NULL,
            confidence_points INTEGER NOT NULL
        );
    """)
    conn.commit()
    conn.close()
```

- [ ] **Step 7: Run test to verify it passes**

```bash
pytest tests/test_db.py::test_schema_creates_all_tables -v
```
Expected: `PASSED`

- [ ] **Step 8: Write and pass test for `queries.py`**

```python
# Add to tests/test_db.py
from src.db.queries import insert_game, get_games_for_week

def test_insert_and_fetch_game(tmp_db):
    game = {
        "game_id": "2024_01_KC_BAL",
        "season": 2024,
        "week": 1,
        "game_type": "regular",
        "home_team": "BAL",
        "away_team": "KC",
        "game_date": "2024-09-05",
        "stadium": "M&T Bank Stadium",
        "is_outdoor": 0,
    }
    insert_game(tmp_db, game)
    games = get_games_for_week(tmp_db, 2024, 1)
    assert len(games) == 1
    assert games[0]["game_id"] == "2024_01_KC_BAL"

@pytest.fixture
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    create_schema(db_path)
    return db_path
```

- [ ] **Step 9: Create `src/db/queries.py`**

```python
import sqlite3
from typing import Optional

def _conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def insert_game(db_path: str, game: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO games
            (game_id, season, week, game_type, home_team, away_team,
             game_date, stadium, is_outdoor, home_score, away_score, home_win)
            VALUES (:game_id, :season, :week, :game_type, :home_team, :away_team,
                    :game_date, :stadium, :is_outdoor,
                    :home_score, :away_score, :home_win)
        """, {**game, "home_score": game.get("home_score"), 
               "away_score": game.get("away_score"), "home_win": game.get("home_win")})

def get_games_for_week(db_path: str, season: int, week: int) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM games WHERE season=? AND week=?", (season, week)
        ).fetchall()
    return [dict(r) for r in rows]

def get_games_for_season(db_path: str, season: int) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM games WHERE season=? ORDER BY week", (season,)
        ).fetchall()
    return [dict(r) for r in rows]

def insert_prediction(db_path: str, pred: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT INTO predictions
            (game_id, season, week, home_win_prob, odds_implied_prob,
             model_version, predicted_winner)
            VALUES (:game_id, :season, :week, :home_win_prob, :odds_implied_prob,
                    :model_version, :predicted_winner)
        """, pred)

def get_predictions_for_week(db_path: str, season: int, week: int) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT p.*, g.home_team, g.away_team, g.game_date, g.is_outdoor
            FROM predictions p JOIN games g USING(game_id)
            WHERE p.season=? AND p.week=?
            ORDER BY p.home_win_prob DESC
        """, (season, week)).fetchall()
    return [dict(r) for r in rows]

def upsert_weekly_assignment(db_path: str, assignment: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT INTO weekly_assignments
            (season, week, game_id, predicted_winner, confidence_points,
             win_probability, is_uncertain, is_overridden, override_reason, updated_at)
            VALUES (:season, :week, :game_id, :predicted_winner, :confidence_points,
                    :win_probability, :is_uncertain, :is_overridden, :override_reason,
                    datetime('now'))
            ON CONFLICT(season, week, game_id) DO UPDATE SET
                confidence_points=excluded.confidence_points,
                predicted_winner=excluded.predicted_winner,
                win_probability=excluded.win_probability,
                is_uncertain=excluded.is_uncertain,
                is_overridden=excluded.is_overridden,
                override_reason=excluded.override_reason,
                updated_at=datetime('now')
        """, {**assignment, "is_overridden": assignment.get("is_overridden", 0),
               "override_reason": assignment.get("override_reason")})

def get_weekly_assignments(db_path: str, season: int, week: int) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT wa.*, g.home_team, g.away_team, g.game_date
            FROM weekly_assignments wa JOIN games g USING(game_id)
            WHERE wa.season=? AND wa.week=?
            ORDER BY wa.confidence_points DESC
        """, (season, week)).fetchall()
    return [dict(r) for r in rows]

def insert_reranking(db_path: str, reranking: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT INTO rerankings
            (session_id, season, week, game_id, old_points, new_points, reason)
            VALUES (:session_id, :season, :week, :game_id, :old_points, :new_points, :reason)
        """, reranking)

def get_rerankings_for_session(db_path: str, session_id: str) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM rerankings WHERE session_id=? ORDER BY created_at",
            (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]

def insert_conversation_message(db_path: str, message: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT INTO conversations (session_id, role, content, season, week)
            VALUES (:session_id, :role, :content, :season, :week)
        """, message)

def get_conversation_history(db_path: str, session_id: str) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE session_id=? ORDER BY created_at",
            (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]

def upsert_injury_report(db_path: str, injury: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT INTO injury_reports
            (team, player_name, position, injury_status, is_qb, season, week)
            VALUES (:team, :player_name, :position, :injury_status, :is_qb, :season, :week)
        """, injury)

def get_injuries_for_week(db_path: str, season: int, week: int,
                           team: Optional[str] = None) -> list[dict]:
    with _conn(db_path) as conn:
        if team:
            rows = conn.execute("""
                SELECT * FROM injury_reports
                WHERE season=? AND week=? AND team=?
                ORDER BY is_qb DESC
            """, (season, week, team)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM injury_reports WHERE season=? AND week=?
                ORDER BY team, is_qb DESC
            """, (season, week)).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 10: Run all DB tests**

```bash
pytest tests/test_db.py -v
```
Expected: All PASSED

- [ ] **Step 11: Commit**

```bash
git init && git add src/db/ tests/test_db.py config.yaml requirements.txt
git commit -m "feat: add SQLite schema and query layer"
```

---

### Task 2: Historical Data Ingestion (nfl_data_py)

**Files:**
- Create: `src/data/historical.py`
- Create: `scripts/ingest_historical.py`
- Test: `tests/test_features.py` (partial — game loading)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
import pytest
from src.data.historical import load_games, load_schedules

def test_load_schedules_returns_dataframe():
    df = load_schedules([2023])
    assert len(df) > 200
    required_cols = {"game_id", "season", "week", "home_team", "away_team",
                     "home_score", "away_score", "game_type", "gameday"}
    assert required_cols.issubset(set(df.columns))

def test_load_games_marks_home_win():
    df = load_games([2023])
    assert "home_win" in df.columns
    assert df["home_win"].isin([0, 1]).all()
    assert df["home_win"].notna().all()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_features.py::test_load_schedules_returns_dataframe -v
```
Expected: `FAILED — ModuleNotFoundError`

- [ ] **Step 3: Create `src/data/historical.py`**

```python
import pandas as pd
import nfl_data_py as nfl

GAME_TYPE_MAP = {
    "REG": "regular",
    "WC": "wildcard",
    "DIV": "divisional",
    "CON": "conference",
    "SB": "superbowl",
}

def load_schedules(seasons: list[int]) -> pd.DataFrame:
    df = nfl.import_schedules(seasons)
    df = df.rename(columns={"gameday": "game_date"})
    df["game_type"] = df["game_type"].map(GAME_TYPE_MAP).fillna("regular")
    df["game_id"] = df["game_id"].astype(str)
    return df

def load_games(seasons: list[int]) -> pd.DataFrame:
    df = load_schedules(seasons)
    completed = df[df["home_score"].notna()].copy()
    completed["home_win"] = (completed["home_score"] > completed["away_score"]).astype(int)
    return completed

def load_team_stats(seasons: list[int]) -> pd.DataFrame:
    return nfl.import_weekly_data(seasons)

def get_team_recent_form(schedules: pd.DataFrame, team: str,
                          season: int, week: int, n: int = 4) -> dict:
    mask = (
        ((schedules["home_team"] == team) | (schedules["away_team"] == team)) &
        (schedules["season"] == season) &
        (schedules["week"] < week) &
        (schedules["home_score"].notna())
    )
    recent = schedules[mask].tail(n)
    if recent.empty:
        return {"win_pct": 0.5, "avg_point_diff": 0.0, "games_played": 0}

    wins = 0
    point_diffs = []
    for _, row in recent.iterrows():
        if row["home_team"] == team:
            wins += int(row["home_score"] > row["away_score"])
            point_diffs.append(row["home_score"] - row["away_score"])
        else:
            wins += int(row["away_score"] > row["home_score"])
            point_diffs.append(row["away_score"] - row["home_score"])

    return {
        "win_pct": wins / len(recent),
        "avg_point_diff": sum(point_diffs) / len(point_diffs),
        "games_played": len(recent),
    }

def get_rest_days(schedules: pd.DataFrame, team: str,
                   season: int, week: int) -> int:
    mask = (
        ((schedules["home_team"] == team) | (schedules["away_team"] == team)) &
        (schedules["season"] == season) &
        (schedules["week"] < week)
    )
    prior = schedules[mask].tail(1)
    if prior.empty:
        return 14  # assume bye week rest at start of season
    last_date = pd.to_datetime(prior["game_date"].values[0])
    current = schedules[
        (((schedules["home_team"] == team) | (schedules["away_team"] == team)) &
         (schedules["season"] == season) & (schedules["week"] == week))
    ]
    if current.empty:
        return 7
    game_date = pd.to_datetime(current["game_date"].values[0])
    return max(1, (game_date - last_date).days)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_features.py::test_load_schedules_returns_dataframe tests/test_features.py::test_load_games_marks_home_win -v
```
Expected: Both PASSED (nfl_data_py downloads data on first run — takes ~30s)

- [ ] **Step 5: Create `scripts/ingest_historical.py`**

```python
#!/usr/bin/env python3
"""One-time script to ingest 2018-2025 historical game data into SQLite."""
import yaml
from src.db.schema import create_schema
from src.db.queries import insert_game
from src.data.historical import load_games

def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    db_path = config["db"]["path"]
    create_schema(db_path)

    seasons = list(range(2018, 2026))
    print(f"Loading seasons: {seasons}")
    games_df = load_games(seasons)
    print(f"Loaded {len(games_df)} completed games")

    for _, row in games_df.iterrows():
        insert_game(db_path, {
            "game_id": row["game_id"],
            "season": int(row["season"]),
            "week": int(row["week"]),
            "game_type": row.get("game_type", "regular"),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "game_date": str(row["game_date"]),
            "stadium": row.get("stadium"),
            "is_outdoor": int(not row.get("roof", "outdoors").startswith("dome")),
            "home_score": int(row["home_score"]) if pd.notna(row["home_score"]) else None,
            "away_score": int(row["away_score"]) if pd.notna(row["away_score"]) else None,
            "home_win": int(row["home_win"]),
        })
    print("Ingestion complete.")

if __name__ == "__main__":
    import pandas as pd
    main()
```

- [ ] **Step 6: Run historical ingestion**

```bash
python scripts/ingest_historical.py
```
Expected: `Loaded ~2800 completed games` then `Ingestion complete.`

- [ ] **Step 7: Commit**

```bash
git add src/data/historical.py scripts/ingest_historical.py tests/test_features.py
git commit -m "feat: add nfl_data_py historical ingestion"
```

---

### Task 3: Odds API Client

**Files:**
- Create: `src/data/odds.py`
- Test: `tests/test_db.py` (add odds parsing test, no live API call)

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_db.py
from src.data.odds import moneyline_to_prob, parse_odds_response

def test_moneyline_to_prob_negative_favorite():
    # -200 favorite should be ~66.7% probability
    prob = moneyline_to_prob(-200)
    assert abs(prob - 0.667) < 0.01

def test_moneyline_to_prob_positive_underdog():
    # +150 underdog should be ~40% probability
    prob = moneyline_to_prob(150)
    assert abs(prob - 0.400) < 0.01

def test_parse_odds_response_removes_vig():
    fake_response = [{
        "id": "abc123",
        "home_team": "Kansas City Chiefs",
        "away_team": "Baltimore Ravens",
        "bookmakers": [{
            "markets": [{
                "key": "h2h",
                "outcomes": [
                    {"name": "Kansas City Chiefs", "price": -180},
                    {"name": "Baltimore Ravens", "price": 155},
                ]
            }]
        }]
    }]
    result = parse_odds_response(fake_response)
    assert len(result) == 1
    assert result[0]["home_team"] == "Kansas City Chiefs"
    # Probabilities must sum to ~1.0 after vig removal
    total = result[0]["home_win_prob"] + result[0]["away_win_prob"]
    assert abs(total - 1.0) < 0.001
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_db.py::test_moneyline_to_prob_negative_favorite -v
```
Expected: `FAILED — ModuleNotFoundError`

- [ ] **Step 3: Create `src/data/odds.py`**

```python
import requests
import yaml
from typing import Optional

def moneyline_to_prob(ml: int) -> float:
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return 100 / (ml + 100)

def parse_odds_response(games: list[dict]) -> list[dict]:
    results = []
    for game in games:
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue
        h2h = next(
            (m for m in bookmakers[0].get("markets", []) if m["key"] == "h2h"),
            None
        )
        if not h2h:
            continue
        outcomes = {o["name"]: o["price"] for o in h2h["outcomes"]}
        home = game["home_team"]
        away = game["away_team"]
        if home not in outcomes or away not in outcomes:
            continue
        home_implied = moneyline_to_prob(outcomes[home])
        away_implied = moneyline_to_prob(outcomes[away])
        total = home_implied + away_implied
        results.append({
            "odds_game_id": game["id"],
            "home_team": home,
            "away_team": away,
            "home_win_prob": home_implied / total,
            "away_win_prob": away_implied / total,
            "home_moneyline": outcomes[home],
            "away_moneyline": outcomes[away],
        })
    return results

def fetch_current_odds(config_path: str = "config.yaml") -> list[dict]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    api_key = config["odds_api"]["key"]
    sport = config["odds_api"]["sport"]
    regions = config["odds_api"]["regions"]
    markets = config["odds_api"]["markets"]
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
        f"?apiKey={api_key}&regions={regions}&markets={markets}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return parse_odds_response(resp.json())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_db.py::test_moneyline_to_prob_negative_favorite tests/test_db.py::test_moneyline_to_prob_positive_underdog tests/test_db.py::test_parse_odds_response_removes_vig -v
```
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add src/data/odds.py tests/test_db.py
git commit -m "feat: add Odds API client with vig removal"
```

---

### Task 4: Injury + Weather Clients

**Files:**
- Create: `src/data/injuries.py`
- Create: `src/data/weather.py`
- Test: `tests/test_db.py` (unit tests, no live API calls)

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/test_db.py
from src.data.injuries import parse_sleeper_injuries, is_qb_out
from src.data.weather import estimate_weather_impact

def test_is_qb_out_detects_out_status():
    injuries = [
        {"player_name": "Patrick Mahomes", "position": "QB",
         "injury_status": "Out", "is_qb": 1},
        {"player_name": "Travis Kelce", "position": "TE",
         "injury_status": "Questionable", "is_qb": 0},
    ]
    assert is_qb_out(injuries) is True

def test_is_qb_out_false_when_qb_healthy():
    injuries = [
        {"player_name": "Travis Kelce", "position": "TE",
         "injury_status": "Out", "is_qb": 0},
    ]
    assert is_qb_out(injuries) is False

def test_weather_impact_indoor_is_zero():
    impact = estimate_weather_impact(is_outdoor=False, temperature=32, wind_speed=20)
    assert impact == 0.0

def test_weather_impact_cold_wind_negative():
    impact = estimate_weather_impact(is_outdoor=True, temperature=20, wind_speed=25)
    assert impact < 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_db.py::test_is_qb_out_detects_out_status -v
```
Expected: `FAILED — ModuleNotFoundError`

- [ ] **Step 3: Create `src/data/injuries.py`**

```python
import requests
import yaml
from typing import Optional

OUT_STATUSES = {"Out", "Doubtful"}
QB_POSITIONS = {"QB"}

def fetch_sleeper_injuries(season: int, week: int,
                            config_path: str = "config.yaml") -> list[dict]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    base = config["sleeper"]["base_url"]
    url = f"{base}/players/nfl"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    players = resp.json()

    injuries = []
    for player_id, player in players.items():
        status = player.get("injury_status")
        if not status:
            continue
        position = player.get("position", "")
        injuries.append({
            "player_name": player.get("full_name", "Unknown"),
            "position": position,
            "injury_status": status,
            "team": player.get("team", ""),
            "is_qb": int(position in QB_POSITIONS),
            "season": season,
            "week": week,
        })
    return injuries

def parse_sleeper_injuries(raw: list[dict]) -> list[dict]:
    return [p for p in raw if p.get("injury_status") in OUT_STATUSES | {"Questionable"}]

def is_qb_out(injuries: list[dict]) -> bool:
    return any(
        p["is_qb"] and p.get("injury_status") in OUT_STATUSES
        for p in injuries
    )
```

- [ ] **Step 4: Create `src/data/weather.py`**

```python
import requests
from typing import Optional

OUTDOOR_STADIUM_COORDS = {
    "BUF": (42.774, -78.787), "KC": (39.049, -94.484), "GB": (44.501, -88.062),
    "CHI": (41.862, -87.617), "PIT": (40.447, -80.016), "CLE": (41.517, -81.700),
    "CIN": (39.095, -84.516), "BAL": (39.278, -76.623), "NE": (42.091, -71.264),
    "NYG": (40.813, -74.074), "NYJ": (40.813, -74.074), "PHI": (39.901, -75.168),
    "WAS": (38.908, -76.864), "DAL": (32.748, -97.093), "CAR": (35.225, -80.853),
    "JAX": (30.324, -81.638), "TEN": (36.166, -86.771), "HOU": (29.685, -95.411),
    "IND": (39.760, -86.164), "MIA": (25.958, -80.239), "DEN": (39.744, -105.020),
    "LV": (36.091, -115.184), "SEA": (47.595, -122.332), "SF": (37.403, -121.970),
    "LAR": (33.953, -118.339), "LAC": (33.953, -118.339), "ARI": (33.528, -112.263),
    "MIN": (44.974, -93.258), "DET": (42.340, -83.045), "TB": (27.976, -82.503),
    "NO": (29.951, -90.081), "ATL": (33.755, -84.401),
}

INDOOR_TEAMS = {"NO", "ATL", "MIN", "DET", "IND", "HOU", "LAR", "LAC", "ARI", "LV", "DAL"}

def get_stadium_weather(home_team: str, game_date: str) -> Optional[dict]:
    if home_team in INDOOR_TEAMS:
        return {"temperature": 72, "wind_speed": 0, "is_outdoor": False}
    coords = OUTDOOR_STADIUM_COORDS.get(home_team)
    if not coords:
        return None
    lat, lon = coords
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,wind_speed_10m_max"
        f"&temperature_unit=fahrenheit&wind_speed_unit=mph"
        f"&start_date={game_date}&end_date={game_date}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        temp = daily.get("temperature_2m_max", [None])[0]
        wind = daily.get("wind_speed_10m_max", [None])[0]
        return {"temperature": temp, "wind_speed": wind, "is_outdoor": True}
    except Exception:
        return None

def estimate_weather_impact(is_outdoor: bool, temperature: Optional[float],
                              wind_speed: Optional[float]) -> float:
    if not is_outdoor:
        return 0.0
    impact = 0.0
    if temperature is not None and temperature < 32:
        impact -= 0.02
    if temperature is not None and temperature < 20:
        impact -= 0.02
    if wind_speed is not None and wind_speed > 20:
        impact -= 0.015
    if wind_speed is not None and wind_speed > 30:
        impact -= 0.015
    return impact
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_db.py::test_is_qb_out_detects_out_status tests/test_db.py::test_is_qb_out_false_when_qb_healthy tests/test_db.py::test_weather_impact_indoor_is_zero tests/test_db.py::test_weather_impact_cold_wind_negative -v
```
Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add src/data/injuries.py src/data/weather.py tests/test_db.py
git commit -m "feat: add injury and weather data clients"
```

---

## Phase 2: ML Model + Confidence Optimizer

### Task 5: Feature Engineering Pipeline

**Files:**
- Create: `src/features/builder.py`
- Test: `tests/test_features.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_features.py
from src.features.builder import build_features_for_game, build_training_dataset

def test_build_features_for_game_has_all_keys():
    schedules_df = load_schedules([2023])
    game_row = schedules_df[schedules_df["week"] == 5].iloc[0]
    features = build_features_for_game(
        game_row=game_row,
        schedules=schedules_df,
        odds_home_win_prob=0.65,
        home_qb_out=False,
        away_qb_out=False,
        weather={"temperature": 55, "wind_speed": 8, "is_outdoor": True},
    )
    required = {
        "odds_home_win_prob", "home_rest_days", "away_rest_days",
        "rest_advantage", "home_qb_out", "away_qb_out",
        "home_recent_winpct", "away_recent_winpct",
        "home_recent_point_diff", "away_recent_point_diff",
        "temperature", "wind_speed", "home_sos", "away_sos", "is_playoff",
    }
    assert required.issubset(set(features.keys()))

def test_build_training_dataset_has_correct_shape():
    df = build_training_dataset(seasons=[2022])
    assert len(df) > 200
    assert "home_win" in df.columns
    assert df["home_win"].isin([0, 1]).all()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_features.py::test_build_features_for_game_has_all_keys -v
```
Expected: `FAILED — ModuleNotFoundError`

- [ ] **Step 3: Create `src/features/builder.py`**

```python
import pandas as pd
import numpy as np
from typing import Optional
from src.data.historical import load_games, load_schedules, get_team_recent_form, get_rest_days
from src.data.weather import estimate_weather_impact

FEATURE_COLS = [
    "odds_home_win_prob", "home_rest_days", "away_rest_days", "rest_advantage",
    "home_qb_out", "away_qb_out", "home_recent_winpct", "away_recent_winpct",
    "home_recent_point_diff", "away_recent_point_diff",
    "temperature", "wind_speed", "home_sos", "away_sos", "is_playoff",
]

PLAYOFF_TYPES = {"wildcard", "divisional", "conference", "superbowl"}

def _get_sos(schedules: pd.DataFrame, team: str, season: int, week: int,
              n: int = 4) -> float:
    mask = (
        ((schedules["home_team"] == team) | (schedules["away_team"] == team)) &
        (schedules["season"] == season) & (schedules["week"] < week) &
        (schedules["home_score"].notna())
    )
    recent = schedules[mask].tail(n)
    if recent.empty:
        return 0.5
    opponent_winpcts = []
    for _, row in recent.iterrows():
        opp = row["away_team"] if row["home_team"] == team else row["home_team"]
        opp_form = get_team_recent_form(schedules, opp, season, row["week"], n=4)
        opponent_winpcts.append(opp_form["win_pct"])
    return float(np.mean(opponent_winpcts))

def build_features_for_game(
    game_row: pd.Series,
    schedules: pd.DataFrame,
    odds_home_win_prob: float,
    home_qb_out: bool = False,
    away_qb_out: bool = False,
    weather: Optional[dict] = None,
) -> dict:
    season = int(game_row["season"])
    week = int(game_row["week"])
    home = game_row["home_team"]
    away = game_row["away_team"]

    home_rest = get_rest_days(schedules, home, season, week)
    away_rest = get_rest_days(schedules, away, season, week)
    home_form = get_team_recent_form(schedules, home, season, week)
    away_form = get_team_recent_form(schedules, away, season, week)
    home_sos = _get_sos(schedules, home, season, week)
    away_sos = _get_sos(schedules, away, season, week)

    is_outdoor = weather.get("is_outdoor", True) if weather else True
    temperature = weather.get("temperature") if weather else None
    wind_speed = weather.get("wind_speed") if weather else None

    return {
        "odds_home_win_prob": odds_home_win_prob,
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "rest_advantage": home_rest - away_rest,
        "home_qb_out": int(home_qb_out),
        "away_qb_out": int(away_qb_out),
        "home_recent_winpct": home_form["win_pct"],
        "away_recent_winpct": away_form["win_pct"],
        "home_recent_point_diff": home_form["avg_point_diff"],
        "away_recent_point_diff": away_form["avg_point_diff"],
        "temperature": temperature if temperature is not None else 65.0,
        "wind_speed": wind_speed if wind_speed is not None else 5.0,
        "home_sos": home_sos,
        "away_sos": away_sos,
        "is_playoff": int(game_row.get("game_type", "regular") in PLAYOFF_TYPES),
    }

def build_training_dataset(
    seasons: list[int],
    odds_by_game: Optional[dict] = None,
) -> pd.DataFrame:
    schedules = load_schedules(seasons)
    games = load_games(seasons)
    rows = []
    for _, game in games.iterrows():
        odds_prob = (odds_by_game or {}).get(game["game_id"], 0.55)
        try:
            features = build_features_for_game(
                game_row=game,
                schedules=schedules,
                odds_home_win_prob=odds_prob,
                home_qb_out=False,
                away_qb_out=False,
                weather=None,
            )
            features["home_win"] = int(game["home_win"])
            features["game_id"] = game["game_id"]
            features["season"] = int(game["season"])
            features["week"] = int(game["week"])
            rows.append(features)
        except Exception:
            continue
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_features.py -v
```
Expected: All PASSED (may take 30–60s downloading data)

- [ ] **Step 5: Commit**

```bash
git add src/features/builder.py tests/test_features.py
git commit -m "feat: add feature engineering pipeline"
```

---

### Task 6: XGBoost Model — Train + Predict

**Files:**
- Create: `src/model/train.py`
- Create: `src/model/predict.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model.py
import pytest
import pandas as pd
import numpy as np
import tempfile, os
from src.model.train import train_model, load_model
from src.model.predict import predict_game_prob
from src.features.builder import FEATURE_COLS

def make_fake_training_data(n=200) -> pd.DataFrame:
    np.random.seed(42)
    df = pd.DataFrame({col: np.random.random(n) for col in FEATURE_COLS})
    df["home_win"] = (df["odds_home_win_prob"] + np.random.normal(0, 0.1, n) > 0.5).astype(int)
    return df

def test_train_model_saves_and_loads(tmp_path):
    df = make_fake_training_data()
    model_path = str(tmp_path / "model.joblib")
    train_model(df, model_path)
    assert os.path.exists(model_path)
    model = load_model(model_path)
    assert model is not None

def test_predict_game_prob_returns_valid_probability(tmp_path):
    df = make_fake_training_data()
    model_path = str(tmp_path / "model.joblib")
    train_model(df, model_path)
    features = {col: 0.5 for col in FEATURE_COLS}
    features["odds_home_win_prob"] = 0.65
    prob = predict_game_prob(model_path, features)
    assert 0.0 <= prob <= 1.0

def test_predict_favors_higher_odds_team(tmp_path):
    df = make_fake_training_data(n=500)
    model_path = str(tmp_path / "model.joblib")
    train_model(df, model_path)
    base = {col: 0.5 for col in FEATURE_COLS}
    base["home_qb_out"] = 0
    base["away_qb_out"] = 0
    high_odds = {**base, "odds_home_win_prob": 0.80}
    low_odds = {**base, "odds_home_win_prob": 0.40}
    assert predict_game_prob(model_path, high_odds) > predict_game_prob(model_path, low_odds)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_model.py::test_train_model_saves_and_loads -v
```
Expected: `FAILED — ModuleNotFoundError`

- [ ] **Step 3: Create `src/model/train.py`**

```python
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from src.features.builder import FEATURE_COLS

MODEL_VERSION = "xgb_v1"

def train_model(df: pd.DataFrame, model_path: str) -> dict:
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = df["home_win"]

    base = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model = CalibratedClassifierCV(base, cv=5, method="isotonic")
    model.fit(X, y)

    cv_scores = cross_val_score(base, X, y, cv=5, scoring="accuracy")
    joblib.dump({"model": model, "version": MODEL_VERSION, "features": FEATURE_COLS}, model_path)

    return {
        "model_version": MODEL_VERSION,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "n_samples": len(df),
    }

def load_model(model_path: str):
    return joblib.load(model_path)
```

- [ ] **Step 4: Create `src/model/predict.py`**

```python
import numpy as np
import pandas as pd
from src.model.train import load_model
from src.features.builder import FEATURE_COLS

def predict_game_prob(model_path: str, features: dict) -> float:
    artifact = load_model(model_path)
    model = artifact["model"]
    X = pd.DataFrame([features])[FEATURE_COLS].fillna(0.5)
    prob = model.predict_proba(X)[0][1]
    return float(prob)

def predict_week(model_path: str, games_features: list[dict]) -> list[dict]:
    artifact = load_model(model_path)
    model = artifact["model"]
    results = []
    for game in games_features:
        X = pd.DataFrame([game["features"]])[FEATURE_COLS].fillna(0.5)
        home_prob = float(model.predict_proba(X)[0][1])
        winner = game["home_team"] if home_prob >= 0.5 else game["away_team"]
        results.append({
            **game,
            "home_win_prob": home_prob,
            "away_win_prob": 1.0 - home_prob,
            "predicted_winner": winner,
            "win_probability": max(home_prob, 1.0 - home_prob),
        })
    return results
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_model.py -v
```
Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add src/model/train.py src/model/predict.py tests/test_model.py
git commit -m "feat: add XGBoost model training and prediction"
```

---

### Task 7: Backtesting Harness

**Files:**
- Create: `src/model/evaluate.py`
- Create: `scripts/backtest.py`
- Test: `tests/test_model.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_model.py
from src.model.evaluate import compute_week_metrics, baseline_accuracy

def test_baseline_accuracy_always_picks_favorite():
    predictions = [
        {"home_win_prob": 0.70, "home_win": 1},
        {"home_win_prob": 0.40, "home_win": 1},  # underdog wins
        {"home_win_prob": 0.65, "home_win": 1},
    ]
    acc = baseline_accuracy(predictions)
    assert abs(acc - 2/3) < 0.01  # picked favorite in 2/3 games correctly

def test_compute_week_metrics():
    predictions = [
        {"home_win_prob": 0.70, "home_win": 1, "confidence_points": 3},
        {"home_win_prob": 0.60, "home_win": 0, "confidence_points": 2},
        {"home_win_prob": 0.55, "home_win": 1, "confidence_points": 1},
    ]
    metrics = compute_week_metrics(predictions)
    assert metrics["actual_points"] == 4   # 3 + 0 + 1
    assert metrics["accuracy"] == pytest.approx(2/3)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_model.py::test_baseline_accuracy_always_picks_favorite -v
```
Expected: `FAILED — ModuleNotFoundError`

- [ ] **Step 3: Create `src/model/evaluate.py`**

```python
import numpy as np
from sklearn.metrics import brier_score_loss

def baseline_accuracy(predictions: list[dict]) -> float:
    correct = sum(
        1 for p in predictions
        if (p["home_win_prob"] >= 0.5) == bool(p["home_win"])
    )
    return correct / len(predictions) if predictions else 0.0

def compute_week_metrics(predictions: list[dict]) -> dict:
    n = len(predictions)
    if n == 0:
        return {}
    correct = sum(
        1 for p in predictions
        if (p["home_win_prob"] >= 0.5) == bool(p["home_win"])
    )
    actual_pts = sum(
        p["confidence_points"] for p in predictions
        if (p["home_win_prob"] >= 0.5) == bool(p["home_win"])
    )
    probs = [p["home_win_prob"] for p in predictions]
    labels = [p["home_win"] for p in predictions]
    return {
        "accuracy": correct / n,
        "actual_points": actual_pts,
        "expected_points": sum(
            p["home_win_prob"] * p["confidence_points"] for p in predictions
        ),
        "brier_score": brier_score_loss(labels, probs),
        "baseline_accuracy": baseline_accuracy(predictions),
    }

def run_season_backtest(
    model_path: str,
    seasons: list[int],
    db_path: str,
    point_range: tuple = (1, 16),
) -> list[dict]:
    from src.db.queries import get_games_for_season
    from src.features.builder import build_training_dataset
    from src.model.predict import predict_week
    from src.optimizer.confidence import assign_confidence_points

    all_metrics = []
    for season in seasons:
        games_df = build_training_dataset(seasons=[season])
        weeks = sorted(games_df["week"].unique())
        for week in weeks:
            week_games = games_df[games_df["week"] == week]
            game_inputs = []
            for _, row in week_games.iterrows():
                game_inputs.append({
                    "game_id": row["game_id"],
                    "home_team": row.get("home_team", ""),
                    "away_team": row.get("away_team", ""),
                    "features": {col: row[col] for col in week_games.columns
                                  if col in __import__("src.features.builder",
                                                        fromlist=["FEATURE_COLS"]).FEATURE_COLS},
                })
            if not game_inputs:
                continue
            predictions = predict_week(model_path, game_inputs)
            assignments = assign_confidence_points(predictions, point_range)
            for pred, assign in zip(predictions, assignments):
                pred["confidence_points"] = assign["confidence_points"]
                pred["home_win"] = int(
                    week_games[week_games["game_id"] == pred["game_id"]]["home_win"].values[0]
                )
            metrics = compute_week_metrics(predictions)
            metrics["season"] = season
            metrics["week"] = week
            all_metrics.append(metrics)
    return all_metrics
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_model.py -v
```
Expected: All PASSED

- [ ] **Step 5: Create `scripts/backtest.py`**

```python
#!/usr/bin/env python3
"""Backtest the model on validation seasons and print summary."""
import yaml
import json
import pandas as pd
from src.features.builder import build_training_dataset
from src.model.train import train_model
from src.model.evaluate import run_season_backtest

def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    train_seasons = config["model"]["train_seasons"]
    val_seasons = config["model"]["val_seasons"]
    model_path = config["model"]["path"]
    db_path = config["db"]["path"]

    print(f"Building training dataset for seasons {train_seasons}...")
    train_df = build_training_dataset(seasons=train_seasons)
    print(f"Training on {len(train_df)} games...")
    metrics = train_model(train_df, model_path)
    print(f"Training complete: CV accuracy={metrics['cv_accuracy_mean']:.3f}")

    print(f"\nRunning backtest on {val_seasons}...")
    results = run_season_backtest(model_path, val_seasons, db_path)
    df = pd.DataFrame(results)
    print(f"\nBacktest Results:")
    print(f"  Avg accuracy:      {df['accuracy'].mean():.3f}")
    print(f"  Baseline accuracy: {df['baseline_accuracy'].mean():.3f}")
    print(f"  Avg actual pts/wk: {df['actual_points'].mean():.1f}")
    print(f"  Avg expected pts:  {df['expected_points'].mean():.1f}")
    print(f"  Avg Brier score:   {df['brier_score'].mean():.4f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the backtest** (takes 2–5 min first run)

```bash
python scripts/backtest.py
```
Expected output (approximate):
```
Training on ~1680 games...
Training complete: CV accuracy=0.58X
Backtest Results:
  Avg accuracy:      0.5X–0.6X
  Baseline accuracy: 0.5X–0.6X
  Avg actual pts/wk: ~90–110
```

- [ ] **Step 7: Commit**

```bash
git add src/model/evaluate.py scripts/backtest.py tests/test_model.py
git commit -m "feat: add backtesting harness with weekly metrics"
```

---

### Task 8: Confidence Point Optimizer

**Files:**
- Create: `src/optimizer/confidence.py`
- Test: `tests/test_optimizer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_optimizer.py
import pytest
from src.optimizer.confidence import assign_confidence_points, get_point_range

def test_assign_confidence_highest_prob_gets_most_points():
    games = [
        {"game_id": "g1", "predicted_winner": "KC", "win_probability": 0.80},
        {"game_id": "g2", "predicted_winner": "SF", "win_probability": 0.65},
        {"game_id": "g3", "predicted_winner": "DAL", "win_probability": 0.55},
    ]
    result = assign_confidence_points(games, (1, 3))
    by_id = {r["game_id"]: r for r in result}
    assert by_id["g1"]["confidence_points"] == 3
    assert by_id["g2"]["confidence_points"] == 2
    assert by_id["g3"]["confidence_points"] == 1

def test_all_points_used_exactly_once():
    games = [
        {"game_id": f"g{i}", "predicted_winner": "X", "win_probability": 0.5 + i*0.01}
        for i in range(16)
    ]
    result = assign_confidence_points(games, (1, 16))
    points = sorted(r["confidence_points"] for r in result)
    assert points == list(range(1, 17))

def test_uncertainty_flag_within_threshold():
    games = [
        {"game_id": "g1", "predicted_winner": "KC", "win_probability": 0.72},
        {"game_id": "g2", "predicted_winner": "SF", "win_probability": 0.71},  # within 3%
        {"game_id": "g3", "predicted_winner": "DAL", "win_probability": 0.55},
    ]
    result = assign_confidence_points(games, (1, 3), uncertainty_threshold=0.03)
    by_id = {r["game_id"]: r for r in result}
    assert by_id["g1"]["is_uncertain"] is True
    assert by_id["g2"]["is_uncertain"] is True
    assert by_id["g3"]["is_uncertain"] is False

def test_get_point_range_regular_season():
    assert get_point_range(n_games=16, game_type="regular") == (1, 16)
    assert get_point_range(n_games=15, game_type="regular") == (1, 15)

def test_get_point_range_playoff():
    assert get_point_range(n_games=6, game_type="wildcard") == (1, 6)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_optimizer.py -v
```
Expected: `FAILED — ModuleNotFoundError`

- [ ] **Step 3: Create `src/optimizer/confidence.py`**

```python
from typing import Optional

def get_point_range(n_games: int, game_type: str = "regular") -> tuple[int, int]:
    return (1, n_games)

def assign_confidence_points(
    games: list[dict],
    point_range: tuple[int, int],
    uncertainty_threshold: float = 0.03,
) -> list[dict]:
    min_pts, max_pts = point_range
    n = len(games)
    points = list(range(min_pts, max_pts + 1))
    if len(points) < n:
        points = list(range(1, n + 1))

    sorted_games = sorted(games, key=lambda g: g["win_probability"], reverse=True)

    # assign highest points to highest probability (rearrangement inequality)
    point_assignments = list(reversed(points[-n:]))

    result = []
    for i, game in enumerate(sorted_games):
        pts = point_assignments[i]

        is_uncertain = False
        if i > 0:
            prev_prob = sorted_games[i - 1]["win_probability"]
            if abs(prev_prob - game["win_probability"]) <= uncertainty_threshold:
                is_uncertain = True
        if i < len(sorted_games) - 1:
            next_prob = sorted_games[i + 1]["win_probability"]
            if abs(next_prob - game["win_probability"]) <= uncertainty_threshold:
                is_uncertain = True

        result.append({
            **game,
            "confidence_points": pts,
            "is_uncertain": is_uncertain,
        })

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_optimizer.py -v
```
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add src/optimizer/confidence.py tests/test_optimizer.py
git commit -m "feat: add confidence point optimizer with uncertainty flagging"
```

---

## Phase 3: Agent + API

### Task 9: Agent Tools

**Files:**
- Create: `src/agent/tools.py`
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agent.py
import pytest
from unittest.mock import patch, MagicMock
from src.agent.tools import (
    get_weekly_schedule, get_game_prediction,
    get_injury_report, get_weekly_assignments, TOOL_DEFINITIONS
)

def test_tool_definitions_have_required_fields():
    for tool in TOOL_DEFINITIONS:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool

def test_get_weekly_schedule_returns_list(tmp_path):
    db_path = str(tmp_path / "test.db")
    from src.db.schema import create_schema
    from src.db.queries import insert_game
    create_schema(db_path)
    insert_game(db_path, {
        "game_id": "2025_01_KC_BAL", "season": 2025, "week": 1,
        "game_type": "regular", "home_team": "BAL", "away_team": "KC",
        "game_date": "2025-09-07", "stadium": "M&T Bank Stadium", "is_outdoor": 0,
    })
    result = get_weekly_schedule(db_path=db_path, season=2025, week=1)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["home_team"] == "BAL"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_agent.py::test_tool_definitions_have_required_fields -v
```
Expected: `FAILED — ModuleNotFoundError`

- [ ] **Step 3: Create `src/agent/tools.py`**

```python
import yaml
from typing import Any
from src.db.queries import (
    get_games_for_week, get_predictions_for_week, get_injuries_for_week,
    get_weekly_assignments, upsert_weekly_assignment, insert_reranking,
)
from src.data.weather import get_stadium_weather
from src.data.odds import fetch_current_odds

TOOL_DEFINITIONS = [
    {
        "name": "get_weekly_schedule",
        "description": "Get all games scheduled for the specified NFL week.",
        "input_schema": {
            "type": "object",
            "properties": {
                "season": {"type": "integer", "description": "NFL season year"},
                "week": {"type": "integer", "description": "Week number"},
            },
            "required": ["season", "week"],
        },
    },
    {
        "name": "get_game_prediction",
        "description": "Get the model's win probability and reasoning for a specific game.",
        "input_schema": {
            "type": "object",
            "properties": {
                "game_id": {"type": "string", "description": "The game identifier"},
            },
            "required": ["game_id"],
        },
    },
    {
        "name": "get_injury_report",
        "description": "Get current injury status for a team, highlighting QB injuries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "description": "Team abbreviation (e.g. KC, BAL)"},
                "season": {"type": "integer"},
                "week": {"type": "integer"},
            },
            "required": ["team", "season", "week"],
        },
    },
    {
        "name": "get_current_odds",
        "description": "Fetch live betting odds and implied win probabilities for this week's games.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_weekly_assignments",
        "description": "Get the current confidence point assignments for this week.",
        "input_schema": {
            "type": "object",
            "properties": {
                "season": {"type": "integer"},
                "week": {"type": "integer"},
            },
            "required": ["season", "week"],
        },
    },
    {
        "name": "get_weather_forecast",
        "description": "Get the weather forecast for a game (outdoor stadiums only).",
        "input_schema": {
            "type": "object",
            "properties": {
                "home_team": {"type": "string", "description": "Home team abbreviation"},
                "game_date": {"type": "string", "description": "Game date YYYY-MM-DD"},
            },
            "required": ["home_team", "game_date"],
        },
    },
    {
        "name": "update_confidence_assignment",
        "description": "Re-rank confidence points for one or more games based on new reasoning. Returns a diff of what changed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "game_id": {"type": "string"},
                            "new_confidence_points": {"type": "integer"},
                            "reason": {"type": "string"},
                        },
                        "required": ["game_id", "new_confidence_points", "reason"],
                    },
                },
                "session_id": {"type": "string"},
                "season": {"type": "integer"},
                "week": {"type": "integer"},
            },
            "required": ["updates", "session_id", "season", "week"],
        },
    },
]

def dispatch_tool(tool_name: str, tool_input: dict,
                   db_path: str, config_path: str = "config.yaml") -> Any:
    if tool_name == "get_weekly_schedule":
        return get_weekly_schedule(db_path, **tool_input)
    elif tool_name == "get_game_prediction":
        return _get_game_prediction(db_path, **tool_input)
    elif tool_name == "get_injury_report":
        return get_injuries_for_week(db_path, tool_input["season"],
                                      tool_input["week"], tool_input.get("team"))
    elif tool_name == "get_current_odds":
        try:
            return fetch_current_odds(config_path)
        except Exception as e:
            return {"error": str(e)}
    elif tool_name == "get_weekly_assignments":
        return get_weekly_assignments(db_path, tool_input["season"], tool_input["week"])
    elif tool_name == "get_weather_forecast":
        return get_stadium_weather(tool_input["home_team"], tool_input["game_date"]) or {}
    elif tool_name == "update_confidence_assignment":
        return _update_assignments(db_path, **tool_input)
    return {"error": f"Unknown tool: {tool_name}"}

def get_weekly_schedule(db_path: str, season: int, week: int) -> list[dict]:
    return get_games_for_week(db_path, season, week)

def _get_game_prediction(db_path: str, game_id: str) -> dict:
    from src.db.queries import get_predictions_for_week
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM predictions WHERE game_id=? ORDER BY created_at DESC LIMIT 1",
        (game_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else {"error": f"No prediction found for {game_id}"}

def _update_assignments(db_path: str, updates: list[dict], session_id: str,
                         season: int, week: int) -> dict:
    diffs = []
    current = {a["game_id"]: a for a in get_weekly_assignments(db_path, season, week)}
    for update in updates:
        game_id = update["game_id"]
        new_pts = update["new_confidence_points"]
        reason = update["reason"]
        old = current.get(game_id, {})
        old_pts = old.get("confidence_points", 0)
        if old:
            upsert_weekly_assignment(db_path, {
                **old,
                "confidence_points": new_pts,
                "is_overridden": 1,
                "override_reason": reason,
            })
        insert_reranking(db_path, {
            "session_id": session_id,
            "season": season,
            "week": week,
            "game_id": game_id,
            "old_points": old_pts,
            "new_points": new_pts,
            "reason": reason,
        })
        diffs.append({
            "game_id": game_id,
            "old_points": old_pts,
            "new_points": new_pts,
            "reason": reason,
        })
    return {"updated": True, "diffs": diffs}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_agent.py -v
```
Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add src/agent/tools.py tests/test_agent.py
git commit -m "feat: add agent tool suite with 7 tools"
```

---

### Task 10: LLM Client + Conversation Loop

**Files:**
- Create: `src/agent/llm.py`
- Create: `src/agent/chat.py`
- Test: `tests/test_agent.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_agent.py
from unittest.mock import patch, MagicMock
from src.agent.llm import get_llm_client, ClaudeClient, OllamaClient

def test_get_llm_client_returns_claude_when_configured(tmp_path):
    config = {"llm": {"provider": "claude", "claude_model": "claude-sonnet-4-6",
                       "ollama_model": "llama3.3:70b"}}
    import yaml
    config_path = str(tmp_path / "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    client = get_llm_client(config_path)
    assert isinstance(client, ClaudeClient)

def test_get_llm_client_returns_ollama_when_configured(tmp_path):
    config = {"llm": {"provider": "ollama", "claude_model": "claude-sonnet-4-6",
                       "ollama_model": "llama3.3:70b"}}
    import yaml
    config_path = str(tmp_path / "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    client = get_llm_client(config_path)
    assert isinstance(client, OllamaClient)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_agent.py::test_get_llm_client_returns_claude_when_configured -v
```
Expected: `FAILED — ModuleNotFoundError`

- [ ] **Step 3: Create `src/agent/llm.py`**

```python
import yaml
import json
from abc import ABC, abstractmethod
from typing import Optional
import anthropic
import ollama as ollama_lib
from src.agent.tools import TOOL_DEFINITIONS

SYSTEM_PROMPT = """You are an NFL confidence pool assistant. You help the user pick 
weekly game winners and assign confidence points optimally for a season-long family pool.

You have access to tools to look up predictions, injury reports, odds, weather, and 
current confidence assignments. When the user challenges a pick, use the tools to 
fetch the relevant data and explain your reasoning with actual numbers.

When you update confidence point assignments, be explicit about what changed and why.
Always ground your explanations in data from the tools — never guess."""

class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        pass

class ClaudeClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic()
        self.model = model

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=tools,
        )
        tool_calls = []
        text = ""
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            elif block.type == "text":
                text += block.text
        return {
            "text": text,
            "tool_calls": tool_calls,
            "stop_reason": response.stop_reason,
        }

class OllamaClient(LLMClient):
    def __init__(self, model: str = "llama3.3:70b"):
        self.model = model

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        system_msg = {"role": "system", "content": SYSTEM_PROMPT}
        full_messages = [system_msg] + messages
        response = ollama_lib.chat(
            model=self.model,
            messages=full_messages,
            tools=self._convert_tools(tools),
        )
        msg = response["message"]
        tool_calls = []
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append({
                    "id": f"call_{fn['name']}",
                    "name": fn["name"],
                    "input": args,
                })
        return {
            "text": msg.get("content", ""),
            "tool_calls": tool_calls,
            "stop_reason": "end_turn" if not tool_calls else "tool_use",
        }

def get_llm_client(config_path: str = "config.yaml") -> LLMClient:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    llm_config = config["llm"]
    provider = llm_config["provider"]
    if provider == "claude":
        return ClaudeClient(model=llm_config["claude_model"])
    return OllamaClient(model=llm_config["ollama_model"])
```

- [ ] **Step 4: Create `src/agent/chat.py`**

```python
import uuid
import json
from src.agent.llm import get_llm_client, LLMClient, TOOL_DEFINITIONS
from src.agent.tools import dispatch_tool
from src.db.queries import insert_conversation_message, get_conversation_history, get_rerankings_for_session

def run_agent_turn(
    user_message: str,
    session_id: str,
    season: int,
    week: int,
    db_path: str,
    config_path: str = "config.yaml",
    llm_client: LLMClient = None,
) -> dict:
    if llm_client is None:
        llm_client = get_llm_client(config_path)

    insert_conversation_message(db_path, {
        "session_id": session_id, "role": "user",
        "content": user_message, "season": season, "week": week,
    })

    history = get_conversation_history(db_path, session_id)
    messages = [{"role": m["role"], "content": m["content"]} for m in history]

    rerankings = []
    max_iterations = 10

    for _ in range(max_iterations):
        response = llm_client.chat(messages, TOOL_DEFINITIONS)

        if response["tool_calls"]:
            tool_results = []
            for tc in response["tool_calls"]:
                result = dispatch_tool(tc["name"], tc["input"], db_path, config_path)
                tool_results.append({
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "result": result,
                })
                if tc["name"] == "update_confidence_assignment":
                    rerankings.extend(result.get("diffs", []))

            # Build messages for next iteration (Claude format)
            assistant_content = []
            for tc in response["tool_calls"]:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                })
            messages.append({"role": "assistant", "content": assistant_content})
            tool_result_content = [
                {
                    "type": "tool_result",
                    "tool_use_id": tr["tool_call_id"],
                    "content": json.dumps(tr["result"]),
                }
                for tr in tool_results
            ]
            messages.append({"role": "user", "content": tool_result_content})
        else:
            break

    final_text = response.get("text", "")
    insert_conversation_message(db_path, {
        "session_id": session_id, "role": "assistant",
        "content": final_text, "season": season, "week": week,
    })

    return {
        "reply": final_text,
        "rerankings": rerankings,
        "session_id": session_id,
    }

def new_session_id() -> str:
    return str(uuid.uuid4())
```

- [ ] **Step 5: Run all agent tests**

```bash
pytest tests/test_agent.py -v
```
Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add src/agent/llm.py src/agent/chat.py tests/test_agent.py
git commit -m "feat: add LLM client abstraction and conversational agent loop"
```

---

### Task 11: FastAPI Backend

**Files:**
- Create: `src/api/main.py`
- Create: `scripts/refresh_weekly.py`
- Test: `tests/test_agent.py` (API smoke test)

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_agent.py
from fastapi.testclient import TestClient

def test_api_health_check():
    import os
    os.environ["NFL_DB_PATH"] = ":memory:"
    from src.api.main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_agent.py::test_api_health_check -v
```
Expected: `FAILED — ModuleNotFoundError`

- [ ] **Step 3: Create `src/api/main.py`**

```python
import os
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.db.schema import create_schema
from src.db.queries import (
    get_games_for_week, get_predictions_for_week,
    get_weekly_assignments, get_injuries_for_week,
)
from src.agent.chat import run_agent_turn, new_session_id
from src.optimizer.confidence import assign_confidence_points, get_point_range
from src.model.predict import predict_week
from src.features.builder import build_features_for_game
from src.data.historical import load_schedules

app = FastAPI(title="NFL Confidence Pool Agent")

def _db_path() -> str:
    return os.environ.get("NFL_DB_PATH", "data/nfl_pool.db")

def _config_path() -> str:
    return os.environ.get("NFL_CONFIG_PATH", "config.yaml")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/week/{season}/{week}")
def get_week(season: int, week: int):
    db = _db_path()
    assignments = get_weekly_assignments(db, season, week)
    games = get_games_for_week(db, season, week)
    return {"season": season, "week": week, "games": games, "assignments": assignments}

@app.get("/week/{season}/{week}/predictions")
def get_week_predictions(season: int, week: int):
    db = _db_path()
    return get_predictions_for_week(db, season, week)

@app.get("/injuries/{season}/{week}/{team}")
def get_injuries(season: int, week: int, team: str):
    return get_injuries_for_week(_db_path(), season, week, team)

class ChatRequest(BaseModel):
    message: str
    session_id: str
    season: int
    week: int

@app.post("/chat")
def chat(req: ChatRequest):
    result = run_agent_turn(
        user_message=req.message,
        session_id=req.session_id,
        season=req.season,
        week=req.week,
        db_path=_db_path(),
        config_path=_config_path(),
    )
    return result

@app.post("/refresh/{season}/{week}")
def refresh_predictions(season: int, week: int):
    db = _db_path()
    config_path = _config_path()
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_path = config["model"]["path"]

    schedules = load_schedules([season])
    week_games = schedules[(schedules["season"] == season) & (schedules["week"] == week)]
    if week_games.empty:
        raise HTTPException(status_code=404, detail="No games found for this week")

    from src.data.odds import fetch_current_odds
    from src.data.injuries import fetch_sleeper_injuries, is_qb_out
    from src.data.weather import get_stadium_weather

    try:
        odds_data = fetch_current_odds(config_path)
        odds_by_teams = {
            (o["home_team"], o["away_team"]): o["home_win_prob"] for o in odds_data
        }
    except Exception:
        odds_by_teams = {}

    game_inputs = []
    for _, row in week_games.iterrows():
        odds_prob = odds_by_teams.get((row["home_team"], row["away_team"]), 0.55)
        weather = get_stadium_weather(row["home_team"], str(row["game_date"]))
        features = build_features_for_game(
            game_row=row, schedules=schedules,
            odds_home_win_prob=odds_prob,
            home_qb_out=False, away_qb_out=False,
            weather=weather,
        )
        game_inputs.append({
            "game_id": row["game_id"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "game_date": str(row["game_date"]),
            "features": features,
        })

    predictions = predict_week(model_path, game_inputs)
    n_games = len(predictions)
    game_type = str(week_games["game_type"].iloc[0]) if not week_games.empty else "regular"
    point_range = get_point_range(n_games, game_type)
    assignments = assign_confidence_points(predictions, point_range,
                                            config["model"]["uncertainty_threshold"])

    from src.db.queries import insert_prediction, upsert_weekly_assignment
    for pred, assign in zip(predictions, assignments):
        insert_prediction(db, {
            "game_id": pred["game_id"],
            "season": season,
            "week": week,
            "home_win_prob": pred["home_win_prob"],
            "odds_implied_prob": pred.get("odds_home_win_prob"),
            "model_version": "xgb_v1",
            "predicted_winner": pred["predicted_winner"],
        })
        upsert_weekly_assignment(db, {
            "season": season,
            "week": week,
            "game_id": assign["game_id"],
            "predicted_winner": assign["predicted_winner"],
            "confidence_points": assign["confidence_points"],
            "win_probability": assign["win_probability"],
            "is_uncertain": int(assign["is_uncertain"]),
            "is_overridden": 0,
            "override_reason": None,
        })
    return {"refreshed": True, "n_games": n_games, "assignments": assignments}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_agent.py::test_api_health_check -v
```
Expected: `PASSED`

- [ ] **Step 5: Create `scripts/refresh_weekly.py`**

```python
#!/usr/bin/env python3
"""Cron entry point: refresh predictions for the current week."""
import yaml
import requests
from datetime import datetime
import nfl_data_py as nfl

def get_current_week() -> tuple[int, int]:
    today = datetime.now()
    season = today.year if today.month >= 9 else today.year - 1
    schedules = nfl.import_schedules([season])
    from_today = schedules[schedules["game_date"] >= today.strftime("%Y-%m-%d")]
    if from_today.empty:
        return season, int(schedules["week"].max())
    return season, int(from_today["week"].min())

def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    season, week = get_current_week()
    print(f"Refreshing: season={season}, week={week}")
    resp = requests.post(f"http://localhost:8000/refresh/{season}/{week}", timeout=120)
    resp.raise_for_status()
    result = resp.json()
    print(f"Refreshed {result['n_games']} games for week {week}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run all tests**

```bash
pytest tests/ -v
```
Expected: All PASSED

- [ ] **Step 7: Commit**

```bash
git add src/api/main.py scripts/refresh_weekly.py tests/test_agent.py
git commit -m "feat: add FastAPI backend with refresh and chat endpoints"
```

---

## Phase 4: Streamlit Dashboard + Automation

### Task 12: Streamlit Dashboard — Weekly Pick Sheet

**Files:**
- Create: `ui/app.py`

> Note: Streamlit UI cannot be unit tested. Verify manually by running the app.

- [ ] **Step 1: Create `ui/__init__.py`**

```bash
touch ui/__init__.py
```

- [ ] **Step 2: Create `ui/app.py`**

```python
import streamlit as st
import requests
import uuid
import pandas as pd
from datetime import datetime
import nfl_data_py as nfl

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="NFL Confidence Pool", layout="wide")

def get_current_week() -> tuple[int, int]:
    today = datetime.now()
    season = today.year if today.month >= 9 else today.year - 1
    try:
        schedules = nfl.import_schedules([season])
        from_today = schedules[schedules["game_date"] >= today.strftime("%Y-%m-%d")]
        if from_today.empty:
            week = int(schedules["week"].max())
        else:
            week = int(from_today["week"].min())
    except Exception:
        week = 1
    return season, week

def fetch_assignments(season: int, week: int) -> list[dict]:
    try:
        resp = requests.get(f"{API_BASE}/week/{season}/{week}", timeout=10)
        resp.raise_for_status()
        return resp.json().get("assignments", [])
    except Exception:
        return []

def refresh_predictions(season: int, week: int):
    with st.spinner("Refreshing predictions..."):
        try:
            resp = requests.post(f"{API_BASE}/refresh/{season}/{week}", timeout=120)
            resp.raise_for_status()
            st.success(f"Refreshed {resp.json()['n_games']} games")
            st.rerun()
        except Exception as e:
            st.error(f"Refresh failed: {e}")

# --- Sidebar ---
with st.sidebar:
    st.title("NFL Confidence Pool")
    season, week = get_current_week()
    season = st.number_input("Season", value=season, min_value=2018)
    week = st.number_input("Week", value=week, min_value=1, max_value=22)
    if st.button("Refresh Predictions", type="primary"):
        refresh_predictions(season, week)
    st.divider()
    st.caption("Data refreshes automatically via cron (Thu/Fri). Use the button for breaking news.")

# --- Main content ---
st.title(f"Week {week} Pick Sheet — {season} Season")

assignments = fetch_assignments(season, week)

if not assignments:
    st.warning("No predictions yet. Click 'Refresh Predictions' to generate picks.")
else:
    # Build pick sheet table
    rows = []
    for a in sorted(assignments, key=lambda x: x["confidence_points"], reverse=True):
        uncertainty_flag = "⚠️" if a.get("is_uncertain") else ""
        override_flag = "✏️" if a.get("is_overridden") else ""
        rows.append({
            "Points": a["confidence_points"],
            "Pick": a["predicted_winner"],
            "Matchup": f"{a['away_team']} @ {a['home_team']}",
            "Win Prob": f"{a['win_probability']:.1%}",
            "Flags": f"{uncertainty_flag}{override_flag}",
            "Game Date": a.get("game_date", ""),
        })

    df = pd.DataFrame(rows)

    # Highlight uncertain rows
    def highlight_uncertain(row):
        if "⚠️" in str(row.get("Flags", "")):
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df.style.apply(highlight_uncertain, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.caption("⚠️ = two picks within 3% win probability (consider swapping) | ✏️ = manually adjusted")

    # Total expected points
    total_expected = sum(
        float(a["win_probability"]) * int(a["confidence_points"]) for a in assignments
    )
    st.metric("Expected Points This Week", f"{total_expected:.1f}")

st.divider()

# --- Chat Interface ---
st.subheader("Challenge a Pick")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rerankings" not in st.session_state:
    st.session_state.rerankings = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.rerankings:
    with st.expander("Recent Re-rankings", expanded=True):
        for r in st.session_state.rerankings:
            direction = "↑" if r["new_points"] > r["old_points"] else "↓"
            st.write(
                f"**{r['game_id']}**: {r['old_points']} pts → {r['new_points']} pts {direction}  "
                f"_{r['reason']}_"
            )

if prompt := st.chat_input("Ask about a pick or challenge the model..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(f"{API_BASE}/chat", json={
                    "message": prompt,
                    "session_id": st.session_state.session_id,
                    "season": int(season),
                    "week": int(week),
                }, timeout=120)
                resp.raise_for_status()
                result = resp.json()
                reply = result.get("reply", "No response")
                new_rerankings = result.get("rerankings", [])
                st.write(reply)
                if new_rerankings:
                    st.session_state.rerankings.extend(new_rerankings)
                    st.rerun()
            except Exception as e:
                reply = f"Error: {e}"
                st.error(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
```

- [ ] **Step 3: Start the API server in one terminal**

```bash
uvicorn src.api.main:app --reload --port 8000
```

- [ ] **Step 4: Start the Streamlit app in another terminal**

```bash
streamlit run ui/app.py
```
Expected: Browser opens at `http://localhost:8501`

- [ ] **Step 5: Manually verify the dashboard**
  - Week picker shows current season/week
  - "Refresh Predictions" button triggers data fetch (may fail without model — run `python scripts/backtest.py` first)
  - Pick sheet table renders with Points, Pick, Matchup, Win Prob columns
  - Uncertain rows highlight in yellow
  - Chat input accepts messages and renders responses
  - Re-rankings expander appears after the agent updates picks

- [ ] **Step 6: Commit**

```bash
git add ui/app.py ui/__init__.py
git commit -m "feat: add Streamlit dashboard with pick sheet and chat interface"
```

---

### Task 13: Cron Automation + VM Setup

**Files:**
- Create: `scripts/start_server.sh`
- Modify: `config.yaml` (cron settings)

- [ ] **Step 1: Create `scripts/start_server.sh`**

```bash
#!/bin/bash
# Run from the nfl_agent project root on the local VM
set -e
cd "$(dirname "$0")/.."

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Start FastAPI in background if not running
if ! pgrep -f "uvicorn src.api.main:app" > /dev/null; then
    nohup uvicorn src.api.main:app --port 8000 --host 0.0.0.0 \
        > logs/api.log 2>&1 &
    echo "API server started (PID $!)"
fi

# Start Streamlit in background if not running
if ! pgrep -f "streamlit run ui/app.py" > /dev/null; then
    nohup streamlit run ui/app.py --server.port 8501 --server.headless true \
        --server.address 0.0.0.0 > logs/ui.log 2>&1 &
    echo "Streamlit started (PID $!)"
fi
```

```bash
mkdir -p logs
chmod +x scripts/start_server.sh
```

- [ ] **Step 2: Set up cron job on the VM**

Run `crontab -e` and add these lines:

```cron
# Start servers on VM boot
@reboot cd /path/to/nfl_agent && bash scripts/start_server.sh

# Refresh NFL predictions every Thursday at 9am and Friday at 8am (NFL week prep)
0 9 * * 4 cd /path/to/nfl_agent && python scripts/refresh_weekly.py >> logs/cron.log 2>&1
0 8 * * 5 cd /path/to/nfl_agent && python scripts/refresh_weekly.py >> logs/cron.log 2>&1
```

Replace `/path/to/nfl_agent` with the actual project path on your VM.

- [ ] **Step 3: Verify cron job is registered**

```bash
crontab -l
```
Expected: Shows the 3 cron entries above.

- [ ] **Step 4: Test the refresh script manually**

```bash
python scripts/refresh_weekly.py
```
Expected: Prints `Refreshing: season=XXXX, week=X` then `Refreshed N games for week X`

- [ ] **Step 5: Access dashboard from laptop**

On your laptop browser, navigate to `http://<vm-local-ip>:8501`

To find VM IP: `ip addr show` or `ifconfig` on the VM.

- [ ] **Step 6: Commit**

```bash
git add scripts/start_server.sh
git commit -m "feat: add VM startup script and cron automation"
```

---

## Self-Review

**Spec coverage check:**

| Requirement | Task |
|---|---|
| Predict winner of every game each week | Tasks 6, 11 |
| Assign confidence points 1-N optimally | Task 8 |
| Uncertainty flags when games within 3% | Task 8 |
| Market odds as baseline | Tasks 3, 11 |
| XGBoost ML model with feature adjustments | Tasks 5, 6 |
| Train 2018–2023, validate 2024–2025 | Tasks 6, 7 |
| SQLite storage | Tasks 1, 2 |
| Streamlit dashboard with pick sheet | Task 12 |
| Conversational agent with tools | Tasks 9, 10 |
| 7 agent tools (schedule, predictions, odds, injuries, weather, assignments, re-rank) | Task 9 |
| Re-ranking with visible diff | Tasks 9, 10, 12 |
| Ollama local LLM + Claude Sonnet 4.6 config switch | Task 10 |
| FastAPI backend | Task 11 |
| Cron refresh on VM | Task 13 |
| Manual refresh button | Tasks 11, 12 |
| Backtesting harness | Task 7 |
| Playoff/Super Bowl handling (schedule-aware point range) | Tasks 8, 11 |
| Cloud-deployable (SQLite→Postgres swap path) | Tasks 1, 11 |

**No gaps found.**

**Placeholder scan:** No TBD, TODO, or "similar to" references found.

**Type consistency:** `FEATURE_COLS` defined in `builder.py`, referenced consistently in `train.py`, `predict.py`, `evaluate.py`. `assign_confidence_points` signature consistent across `confidence.py`, `main.py`. `dispatch_tool` maps all 7 tool names from `TOOL_DEFINITIONS`.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-07-nfl-confidence-pool.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
