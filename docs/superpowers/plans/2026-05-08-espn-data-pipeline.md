# ESPN Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `nfl_data_py` and Sleeper API with the ESPN unofficial API as the single source of truth for all game data, adding box score stats (turnovers, yards, third-down rate) and structured injury/depth chart data to improve model features.

**Architecture:** Raw `httpx` calls to `site.api.espn.com` populate a redesigned SQLite schema (ESPN numeric event ID as PK). The backfill script iterates seasons 2018–2025 week-by-week, checkpointing on `espn_id` so it can resume. Feature computation is DB-centric — `build_features_for_game` queries SQLite instead of operating on in-memory DataFrames. The Odds API and Open-Meteo are unchanged.

**Tech Stack:** Python 3.11, httpx, SQLite, pandas, XGBoost, pytest

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `src/data/espn.py` | **Create** | ESPN HTTP client + JSON parsers |
| `src/db/schema.py` | **Rewrite** | ESPN-native schema (espn_id PK, team_game_stats, depth_charts) |
| `src/db/queries.py` | **Extend** | New insert/query functions for ESPN tables |
| `src/data/historical.py` | **Rewrite** | DB-backed data access layer (replaces nfl_data_py calls) |
| `src/data/injuries.py` | **Rewrite** | ESPN injury fetcher (replaces Sleeper) |
| `src/features/builder.py` | **Rewrite** | New FEATURE_COLS (27 features), DB-centric feature computation |
| `scripts/ingest_historical.py` | **Rewrite** | ESPN backfill with espn_id checkpointing |
| `src/model/evaluate.py` | **Modify** | `game_id` → `espn_id`, pass `db_path` to build_training_dataset |
| `scripts/backtest.py` | **Modify** | Pass `db_path` to build_training_dataset |
| `pyproject.toml` | **Modify** | Remove nfl_data_py |
| `config.yaml` | **Modify** | Remove sleeper section |
| `tests/test_espn.py` | **Create** | Parse tests using repo fixture files |
| `tests/test_db.py` | **Rewrite** | Updated for new schema |
| `tests/test_features.py` | **Rewrite** | DB-fixture-based feature tests |
| `tests/test_model.py` | **Modify** | Update stale feature name references |

---

## Task 1: Remove nfl_data_py, update config

**Files:**
- Modify: `pyproject.toml`
- Modify: `config.yaml`

- [ ] **Step 1: Remove nfl_data_py from pyproject.toml**

Edit `pyproject.toml` — remove the `"nfl_data_py>=0.2",` line. `httpx` is already present.

Final dependencies block (relevant section):
```toml
dependencies = [
    "xgboost>=2.0",
    "scikit-learn>=1.4",
    "pandas>=2.0",
    "numpy>=1.26",
    "requests>=2.31",
    "openmeteo-requests>=1.2",
    "retry-requests>=2.0",
    "fastapi>=0.111",
    "uvicorn>=0.29",
    "streamlit>=1.35",
    "anthropic>=0.28",
    "ollama>=0.2",
    "joblib>=1.4",
    "pyyaml>=6.0",
    "pytest>=8.0",
    "httpx>=0.27",
]
```

- [ ] **Step 2: Remove sleeper section from config.yaml**

Replace the `sleeper:` block with an ESPN base URL entry:

```yaml
odds_api:
  key: "YOUR_ODDS_API_KEY"
  sport: "americanfootball_nfl"
  regions: "us"
  markets: "h2h"

espn:
  base_url: "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

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

- [ ] **Step 3: Sync dependencies**

```bash
uv sync
```

Expected: resolves without nfl_data_py.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml config.yaml uv.lock
git commit -m "chore: remove nfl_data_py, add espn config"
```

---

## Task 2: Create ESPN HTTP client and parsers

**Files:**
- Create: `src/data/espn.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_espn.py`:

```python
import json
from pathlib import Path
import pytest
from src.data.espn import parse_game, parse_box_score, _parse_split, _parse_possession


def load_season_fixture():
    return json.loads(Path("nfl_season_2020.json").read_text())


def test_parse_split_made_att():
    assert _parse_split("7-14") == (7, 14)
    assert _parse_split("0-0") == (0, 0)
    assert _parse_split("3-4") == (3, 4)


def test_parse_possession_to_seconds():
    assert _parse_possession("32:14") == 32 * 60 + 14
    assert _parse_possession("0:00") == 0
    assert _parse_possession("28:30") == 28 * 60 + 30


def test_parse_game_extracts_teams():
    data = load_season_fixture()
    event = data["events"][0]
    game = parse_game(event)
    assert game["home_team"] == "HOU"
    assert game["away_team"] == "BUF"
    assert game["espn_id"] == event["id"]
    assert game["season"] == 2019  # ESPN year for Jan 2020 playoff game
    assert isinstance(game["is_indoor"], int)
    assert isinstance(game["is_neutral"], int)


def test_parse_game_completed_sets_home_win():
    data = load_season_fixture()
    # First event is a completed playoff game (STATUS_FINAL)
    event = data["events"][0]
    game = parse_game(event)
    assert game["home_win"] in (0, 1)
    assert game["home_score"] is not None
    assert game["away_score"] is not None


def test_parse_game_postseason_type():
    data = load_season_fixture()
    # All events in season fixture are post-season (type=3)
    event = data["events"][0]
    game = parse_game(event)
    assert game["game_type"] in ("wildcard", "divisional", "conference", "superbowl")


def test_parse_box_score_returns_two_teams():
    summary = {
        "boxscore": {
            "teams": [
                {
                    "team": {"abbreviation": "HOU"},
                    "homeAway": "home",
                    "statistics": [
                        {"name": "totalYards", "displayValue": "300"},
                        {"name": "netPassingYards", "displayValue": "200"},
                        {"name": "rushingYards", "displayValue": "100"},
                        {"name": "turnovers", "displayValue": "2"},
                        {"name": "firstDowns", "displayValue": "18"},
                        {"name": "thirdDownEff", "displayValue": "5-13"},
                        {"name": "redZoneAtts", "displayValue": "2-3"},
                        {"name": "possessionTime", "displayValue": "28:30"},
                        {"name": "sacksYardsLost", "displayValue": "3-21"},
                    ],
                },
                {
                    "team": {"abbreviation": "BUF"},
                    "homeAway": "away",
                    "statistics": [
                        {"name": "totalYards", "displayValue": "350"},
                        {"name": "turnovers", "displayValue": "1"},
                        {"name": "thirdDownEff", "displayValue": "7-12"},
                        {"name": "redZoneAtts", "displayValue": "3-4"},
                        {"name": "possessionTime", "displayValue": "31:30"},
                        {"name": "sacksYardsLost", "displayValue": "1-7"},
                    ],
                },
            ]
        }
    }
    results = parse_box_score(summary, "401220225")
    assert len(results) == 2
    hou = next(r for r in results if r["team"] == "HOU")
    assert hou["total_yards"] == 300
    assert hou["turnovers"] == 2
    assert hou["third_down_att"] == 13
    assert hou["third_down_made"] == 5
    assert hou["possession_secs"] == 28 * 60 + 30
    assert hou["sacks_taken"] == 3
    assert hou["is_home"] == 1
    buf = next(r for r in results if r["team"] == "BUF")
    assert buf["is_home"] == 0
    assert buf["total_yards"] == 350


def test_parse_box_score_empty_summary():
    results = parse_box_score({}, "999")
    assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_espn.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `src/data/espn.py` doesn't exist yet.

- [ ] **Step 3: Create `src/data/espn.py`**

```python
import httpx

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

_POSTSEASON_WEEK_TO_TYPE = {
    1: "wildcard",
    2: "divisional",
    3: "conference",
    4: "superbowl",
    5: "superbowl",  # ESPN sometimes uses week 5
}


def _get(path: str, params: dict = None) -> dict:
    url = f"{ESPN_BASE}/{path}"
    with httpx.Client(timeout=30) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def fetch_scoreboard(season: int, week: int, season_type: int = 2) -> list[dict]:
    """Return list of event dicts. season_type: 2=regular, 3=postseason."""
    data = _get("scoreboard", params={"season": season, "week": week, "seasontype": season_type})
    return data.get("events", [])


def fetch_game_summary(espn_id: str) -> dict:
    """Return full game summary including boxscore."""
    return _get("summary", params={"event": espn_id})


def fetch_team_injuries(team_id: str) -> list[dict]:
    """Return current raw injury list for a team."""
    data = _get(f"teams/{team_id}/injuries")
    return data.get("injuries", [])


def fetch_team_depth_chart(team_id: str) -> dict:
    """Return raw depth chart response for a team."""
    return _get(f"teams/{team_id}/depthcharts")


def _parse_split(value: str) -> tuple[int, int]:
    """Parse 'made-att' string like '7-14' → (7, 14)."""
    parts = value.split("-")
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return 0, 0
    return 0, 0


def _parse_possession(value: str) -> int:
    """Parse 'MM:SS' to total seconds."""
    parts = value.split(":")
    if len(parts) == 2:
        try:
            return int(parts[0]) * 60 + int(parts[1])
        except ValueError:
            return 0
    return 0


def parse_game(event: dict) -> dict:
    """Normalize an ESPN scoreboard event to a flat dict for DB insertion."""
    comp = event.get("competitions", [{}])[0]
    competitors = comp.get("competitors", [])
    home = next((c for c in competitors if c.get("homeAway") == "home"), {})
    away = next((c for c in competitors if c.get("homeAway") == "away"), {})

    season_info = event.get("season", {})
    season_type = season_info.get("type", 2)
    week_num = event.get("week", {}).get("number", 0)

    if season_type == 3:
        game_type = _POSTSEASON_WEEK_TO_TYPE.get(week_num, "wildcard")
    else:
        game_type = "regular"

    venue = comp.get("venue", {})
    status_name = event.get("status", {}).get("type", {}).get("name", "")
    is_final = "FINAL" in status_name.upper()

    home_score_raw = home.get("score")
    away_score_raw = away.get("score")

    try:
        home_score = int(float(home_score_raw)) if home_score_raw is not None else None
        away_score = int(float(away_score_raw)) if away_score_raw is not None else None
    except (ValueError, TypeError):
        home_score = away_score = None

    if is_final and home_score is not None and away_score is not None:
        home_win = int(home_score > away_score)
    else:
        home_win = None

    home_team = home.get("team", {})
    away_team = away.get("team", {})

    return {
        "espn_id": str(event["id"]),
        "season": season_info.get("year"),
        "week": week_num,
        "game_type": game_type,
        "home_team": home_team.get("abbreviation", ""),
        "away_team": away_team.get("abbreviation", ""),
        "home_espn_id": str(home_team.get("id", "")),
        "away_espn_id": str(away_team.get("id", "")),
        "game_date": event.get("date", ""),
        "venue": venue.get("fullName", ""),
        "is_indoor": int(venue.get("indoor", False)),
        "is_neutral": int(comp.get("neutralSite", False)),
        "attendance": comp.get("attendance"),
        "home_score": home_score,
        "away_score": away_score,
        "home_win": home_win,
    }


def parse_box_score(summary: dict, espn_id: str) -> list[dict]:
    """Parse ESPN summary boxscore into per-team stat dicts."""
    teams = summary.get("boxscore", {}).get("teams", [])
    results = []
    for team_data in teams:
        team = team_data.get("team", {})
        is_home = int(team_data.get("homeAway", "away") == "home")
        stats = {s["name"]: s.get("displayValue", "0") for s in team_data.get("statistics", [])}

        third_made, third_att = _parse_split(stats.get("thirdDownEff", "0-0"))
        rz_made, rz_att = _parse_split(stats.get("redZoneAtts", "0-0"))
        sacks_taken, _ = _parse_split(stats.get("sacksYardsLost", "0-0"))

        def _int(key: str) -> int:
            try:
                return int(stats.get(key, 0) or 0)
            except (ValueError, TypeError):
                return 0

        results.append({
            "espn_id": espn_id,
            "team": team.get("abbreviation", ""),
            "is_home": is_home,
            "total_yards": _int("totalYards"),
            "pass_yards": _int("netPassingYards"),
            "rush_yards": _int("rushingYards"),
            "turnovers": _int("turnovers"),
            "first_downs": _int("firstDowns"),
            "third_down_att": third_att,
            "third_down_made": third_made,
            "red_zone_att": rz_att,
            "red_zone_made": rz_made,
            "possession_secs": _parse_possession(stats.get("possessionTime", "0:00")),
            "sacks_taken": sacks_taken,
        })
    return results


def parse_game_injuries(summary: dict, season: int, week: int) -> list[dict]:
    """Extract game-day injury data from a summary (for historical backfill)."""
    injuries = []
    for section in summary.get("injuries", []):
        for item in section.get("injuries", []):
            athlete = item.get("athlete", {})
            position = athlete.get("position", {}).get("abbreviation", "")
            team = athlete.get("team", {}).get("abbreviation", "")
            if not team or not athlete.get("id"):
                continue
            injuries.append({
                "season": season,
                "week": week,
                "team": team,
                "athlete_id": str(athlete["id"]),
                "athlete_name": athlete.get("displayName", ""),
                "position": position,
                "status": item.get("status", ""),
                "is_qb": int(position == "QB"),
            })
    return injuries


def parse_depth_chart_qbs(depth_chart: dict, team_abbr: str, season: int, week: int) -> list[dict]:
    """Extract QB depth chart entries (rank 1 = starter)."""
    entries = []
    for item in depth_chart.get("items", []):
        positions = item.get("positions", {})
        qb_data = positions.get("QB", {})
        for athlete_entry in qb_data.get("athletes", []):
            athlete = athlete_entry.get("athlete", {})
            entries.append({
                "season": season,
                "week": week,
                "team": team_abbr,
                "athlete_id": str(athlete.get("id", "")),
                "athlete_name": athlete.get("displayName", ""),
                "rank": int(athlete_entry.get("rank", 99)),
            })
    return entries
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_espn.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/data/espn.py tests/test_espn.py
git commit -m "feat: add ESPN HTTP client and JSON parsers"
```

---

## Task 3: Redesign database schema

**Files:**
- Rewrite: `src/db/schema.py`
- Modify: `tests/test_db.py` (partial — just the schema table list test)

- [ ] **Step 1: Write the failing schema test**

Update `tests/test_db.py` — replace `test_schema_creates_all_tables` with:

```python
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
        expected = {
            "games", "team_game_stats", "injury_reports", "depth_charts",
            "predictions", "weekly_assignments", "conversations",
            "rerankings", "model_metrics", "family_picks",
        }
        assert expected.issubset(tables)
    finally:
        os.unlink(db_path)


@pytest.fixture
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    create_schema(db_path)
    return db_path
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_db.py::test_schema_creates_all_tables -v
```

Expected: FAIL — `team_game_stats` and `depth_charts` don't exist yet.

- [ ] **Step 3: Rewrite `src/db/schema.py`**

```python
import sqlite3


def create_schema(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS games (
            espn_id TEXT PRIMARY KEY,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            game_type TEXT NOT NULL DEFAULT 'regular',
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_espn_id TEXT,
            away_espn_id TEXT,
            game_date TEXT NOT NULL,
            venue TEXT,
            is_indoor INTEGER DEFAULT 0,
            is_neutral INTEGER DEFAULT 0,
            attendance INTEGER,
            home_score INTEGER,
            away_score INTEGER,
            home_win INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS team_game_stats (
            espn_id TEXT NOT NULL,
            team TEXT NOT NULL,
            is_home INTEGER NOT NULL,
            total_yards INTEGER,
            pass_yards INTEGER,
            rush_yards INTEGER,
            turnovers INTEGER,
            first_downs INTEGER,
            third_down_att INTEGER,
            third_down_made INTEGER,
            red_zone_att INTEGER,
            red_zone_made INTEGER,
            possession_secs INTEGER,
            sacks_taken INTEGER,
            PRIMARY KEY (espn_id, team),
            FOREIGN KEY (espn_id) REFERENCES games(espn_id)
        );

        CREATE TABLE IF NOT EXISTS injury_reports (
            injury_id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            team TEXT NOT NULL,
            athlete_id TEXT NOT NULL,
            athlete_name TEXT NOT NULL,
            position TEXT,
            status TEXT,
            is_qb INTEGER DEFAULT 0,
            fetched_at TEXT DEFAULT (datetime('now')),
            UNIQUE(season, week, athlete_id)
        );

        CREATE TABLE IF NOT EXISTS depth_charts (
            depth_chart_id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            team TEXT NOT NULL,
            athlete_id TEXT NOT NULL,
            athlete_name TEXT NOT NULL,
            rank INTEGER NOT NULL,
            fetched_at TEXT DEFAULT (datetime('now')),
            UNIQUE(season, week, team, rank)
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
            FOREIGN KEY (game_id) REFERENCES games(espn_id)
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
    finally:
        conn.close()
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_db.py::test_schema_creates_all_tables -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/db/schema.py tests/test_db.py
git commit -m "feat: redesign schema with ESPN-native tables"
```

---

## Task 4: Add ESPN query functions to queries.py

**Files:**
- Extend: `src/db/queries.py`
- Extend: `tests/test_db.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_db.py`:

```python
def test_insert_and_fetch_espn_game(tmp_db):
    from src.db.queries import insert_espn_game, get_games_for_week
    game = {
        "espn_id": "401220225",
        "season": 2024,
        "week": 1,
        "game_type": "regular",
        "home_team": "BAL",
        "away_team": "KC",
        "home_espn_id": "33",
        "away_espn_id": "12",
        "game_date": "2024-09-05T20:00Z",
        "venue": "M&T Bank Stadium",
        "is_indoor": 0,
        "is_neutral": 0,
        "attendance": 71000,
        "home_score": 27,
        "away_score": 20,
        "home_win": 1,
    }
    insert_espn_game(tmp_db, game)
    games = get_games_for_week(tmp_db, 2024, 1)
    assert len(games) == 1
    assert games[0]["espn_id"] == "401220225"
    assert games[0]["home_win"] == 1


def test_get_existing_espn_ids(tmp_db):
    from src.db.queries import insert_espn_game, get_existing_espn_ids
    game = {
        "espn_id": "999", "season": 2024, "week": 1, "game_type": "regular",
        "home_team": "BAL", "away_team": "KC", "home_espn_id": "33",
        "away_espn_id": "12", "game_date": "2024-09-05T20:00Z",
        "venue": "X", "is_indoor": 0, "is_neutral": 0,
        "attendance": None, "home_score": None, "away_score": None, "home_win": None,
    }
    insert_espn_game(tmp_db, game)
    ids = get_existing_espn_ids(tmp_db)
    assert "999" in ids


def test_insert_and_fetch_team_stats(tmp_db):
    from src.db.queries import insert_espn_game, insert_team_stats, get_team_box_stats
    game = {
        "espn_id": "401220225", "season": 2024, "week": 1, "game_type": "regular",
        "home_team": "BAL", "away_team": "KC", "home_espn_id": "33",
        "away_espn_id": "12", "game_date": "2024-09-05T20:00Z", "venue": "M",
        "is_indoor": 0, "is_neutral": 0, "attendance": 71000,
        "home_score": 27, "away_score": 20, "home_win": 1,
    }
    insert_espn_game(tmp_db, game)
    stats = {
        "espn_id": "401220225", "team": "BAL", "is_home": 1,
        "total_yards": 350, "pass_yards": 220, "rush_yards": 130,
        "turnovers": 1, "first_downs": 22, "third_down_att": 14,
        "third_down_made": 7, "red_zone_att": 4, "red_zone_made": 3,
        "possession_secs": 1920, "sacks_taken": 1,
    }
    insert_team_stats(tmp_db, stats)
    rows = get_team_box_stats(tmp_db, "BAL", 2024, 2)
    assert len(rows) == 1
    assert rows[0]["total_yards"] == 350
    assert rows[0]["turnovers"] == 1


def test_get_team_results_returns_recent_games(tmp_db):
    from src.db.queries import insert_espn_game, get_team_results
    for i, (espn_id, week, home_score, away_score, home_win) in enumerate([
        ("g1", 1, 27, 20, 1),
        ("g2", 2, 14, 28, 0),
        ("g3", 3, 35, 17, 1),
    ]):
        insert_espn_game(tmp_db, {
            "espn_id": espn_id, "season": 2024, "week": week,
            "game_type": "regular", "home_team": "BAL", "away_team": "KC",
            "home_espn_id": "33", "away_espn_id": "12",
            "game_date": f"2024-09-0{week + 4}T20:00Z", "venue": "M",
            "is_indoor": 0, "is_neutral": 0, "attendance": None,
            "home_score": home_score, "away_score": away_score, "home_win": home_win,
        })
    results = get_team_results(tmp_db, "BAL", 2024, before_week=4, n=4)
    assert len(results) == 3
    # Most recent first
    assert results[0]["espn_id"] == "g3"


def test_insert_and_fetch_injury(tmp_db):
    from src.db.queries import insert_injury, get_injuries_for_week
    injury = {
        "season": 2024, "week": 5, "team": "KC",
        "athlete_id": "3139477", "athlete_name": "Patrick Mahomes",
        "position": "QB", "status": "Questionable", "is_qb": 1,
    }
    insert_injury(tmp_db, injury)
    rows = get_injuries_for_week(tmp_db, "KC", 2024, 5)
    assert len(rows) == 1
    assert rows[0]["athlete_name"] == "Patrick Mahomes"
    assert rows[0]["is_qb"] == 1


def test_insert_and_fetch_depth_chart(tmp_db):
    from src.db.queries import insert_depth_chart_entry, get_starting_qb
    entry = {
        "season": 2024, "week": 5, "team": "KC",
        "athlete_id": "3139477", "athlete_name": "Patrick Mahomes", "rank": 1,
    }
    insert_depth_chart_entry(tmp_db, entry)
    qb = get_starting_qb(tmp_db, "KC", 2024, 5)
    assert qb is not None
    assert qb["athlete_name"] == "Patrick Mahomes"
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_db.py -k "espn or team_stats or team_results or injury or depth" -v
```

Expected: all fail with `ImportError` — new functions don't exist yet.

- [ ] **Step 3: Add new functions to `src/db/queries.py`**

Add these functions (keep all existing functions — `insert_prediction`, `upsert_weekly_assignment`, etc. remain unchanged, just remove the old `insert_game` and update `get_games_for_week`/`get_games_for_season`):

```python
import sqlite3
from typing import Optional


def _conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def get_existing_espn_ids(db_path: str) -> set:
    with _conn(db_path) as conn:
        rows = conn.execute("SELECT espn_id FROM games").fetchall()
    return {r[0] for r in rows}


def insert_espn_game(db_path: str, game: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO games
            (espn_id, season, week, game_type, home_team, away_team,
             home_espn_id, away_espn_id, game_date, venue,
             is_indoor, is_neutral, attendance, home_score, away_score, home_win)
            VALUES (:espn_id, :season, :week, :game_type, :home_team, :away_team,
                    :home_espn_id, :away_espn_id, :game_date, :venue,
                    :is_indoor, :is_neutral, :attendance,
                    :home_score, :away_score, :home_win)
        """, game)


def insert_team_stats(db_path: str, stats: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO team_game_stats
            (espn_id, team, is_home, total_yards, pass_yards, rush_yards,
             turnovers, first_downs, third_down_att, third_down_made,
             red_zone_att, red_zone_made, possession_secs, sacks_taken)
            VALUES (:espn_id, :team, :is_home, :total_yards, :pass_yards, :rush_yards,
                    :turnovers, :first_downs, :third_down_att, :third_down_made,
                    :red_zone_att, :red_zone_made, :possession_secs, :sacks_taken)
        """, stats)


def insert_injury(db_path: str, injury: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT OR IGNORE INTO injury_reports
            (season, week, team, athlete_id, athlete_name, position, status, is_qb)
            VALUES (:season, :week, :team, :athlete_id, :athlete_name,
                    :position, :status, :is_qb)
        """, injury)


def insert_depth_chart_entry(db_path: str, entry: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO depth_charts
            (season, week, team, athlete_id, athlete_name, rank)
            VALUES (:season, :week, :team, :athlete_id, :athlete_name, :rank)
        """, entry)


def get_games_for_week(db_path: str, season: int, week: int) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM games WHERE season=? AND week=?", (season, week)
        ).fetchall()
    return [dict(r) for r in rows]


def get_games_for_season(db_path: str, season: int) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM games WHERE season=? ORDER BY game_date", (season,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_team_results(db_path: str, team: str, season: int,
                     before_week: int, n: int = 4) -> list[dict]:
    """Last n completed games for team before before_week. Most recent first."""
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT espn_id, season, week, game_date, home_team, away_team,
                   home_score, away_score, home_win
            FROM games
            WHERE (home_team=? OR away_team=?)
              AND season=? AND week<? AND home_win IS NOT NULL
            ORDER BY game_date DESC
            LIMIT ?
        """, (team, team, season, before_week, n)).fetchall()
    return [dict(r) for r in rows]


def get_team_box_stats(db_path: str, team: str, season: int,
                       before_week: int, n: int = 4) -> list[dict]:
    """Last n box score rows for team before before_week. Most recent first."""
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT t.*
            FROM team_game_stats t
            JOIN games g USING(espn_id)
            WHERE t.team=? AND g.season=? AND g.week<? AND g.home_win IS NOT NULL
            ORDER BY g.game_date DESC
            LIMIT ?
        """, (team, season, before_week, n)).fetchall()
    return [dict(r) for r in rows]


def get_injuries_for_week(db_path: str, team: str, season: int, week: int) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT * FROM injury_reports WHERE team=? AND season=? AND week=?
        """, (team, season, week)).fetchall()
    return [dict(r) for r in rows]


def get_starting_qb(db_path: str, team: str, season: int, week: int) -> Optional[dict]:
    with _conn(db_path) as conn:
        row = conn.execute("""
            SELECT * FROM depth_charts
            WHERE team=? AND season=? AND week=? AND rank=1
        """, (team, season, week)).fetchone()
    return dict(row) if row else None


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
            SELECT p.*, g.home_team, g.away_team, g.game_date, g.is_indoor
            FROM predictions p JOIN games g ON p.game_id = g.espn_id
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
            FROM weekly_assignments wa JOIN games g ON wa.game_id = g.espn_id
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
```

- [ ] **Step 4: Run all DB tests**

```bash
uv run pytest tests/test_db.py -v
```

Expected: all tests pass. (The old `test_insert_and_fetch_game` and `test_is_qb_out_*` tests may fail — they'll be replaced in Task 9.)

- [ ] **Step 5: Commit**

```bash
git add src/db/queries.py tests/test_db.py
git commit -m "feat: add ESPN query functions and update DB tests"
```

---

## Task 5: Rewrite data access layer (historical.py)

**Files:**
- Rewrite: `src/data/historical.py`

- [ ] **Step 1: Rewrite `src/data/historical.py`**

This module now reads from the ESPN-populated SQLite DB. No `nfl_data_py` imports.

```python
"""DB-backed game data access layer. All data sourced from ESPN via SQLite."""
import sqlite3
from typing import Optional
import pandas as pd
from src.db.queries import get_team_results, get_team_box_stats

PLAYOFF_TYPES = {"wildcard", "divisional", "conference", "superbowl"}


def load_games(db_path: str, seasons: list[int]) -> pd.DataFrame:
    """Load completed games from DB for given seasons."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(seasons))
    rows = conn.execute(
        f"SELECT * FROM games WHERE season IN ({placeholders}) AND home_win IS NOT NULL"
        f" ORDER BY game_date",
        seasons,
    ).fetchall()
    conn.close()
    return pd.DataFrame([dict(r) for r in rows])


def load_schedules(db_path: str, seasons: list[int]) -> pd.DataFrame:
    """Load all games (including upcoming) from DB for given seasons."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(seasons))
    rows = conn.execute(
        f"SELECT * FROM games WHERE season IN ({placeholders}) ORDER BY game_date",
        seasons,
    ).fetchall()
    conn.close()
    return pd.DataFrame([dict(r) for r in rows])


def get_team_recent_form(db_path: str, team: str, season: int,
                          week: int, n: int = 4) -> dict:
    results = get_team_results(db_path, team, season, week, n)
    if not results:
        return {"win_pct": 0.5, "avg_point_diff": 0.0, "games_played": 0}
    wins = 0
    point_diffs = []
    for r in results:
        if r["home_team"] == team:
            wins += int(r["home_win"] == 1)
            point_diffs.append(r["home_score"] - r["away_score"])
        else:
            wins += int(r["home_win"] == 0)
            point_diffs.append(r["away_score"] - r["home_score"])
    return {
        "win_pct": wins / len(results),
        "avg_point_diff": sum(point_diffs) / len(point_diffs),
        "games_played": len(results),
    }


def get_home_road_winpct(db_path: str, team: str, season: int,
                          week: int, home_games: bool, n: int = 4) -> float:
    """Win% in home games only or road games only over last n such games."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if home_games:
        rows = conn.execute("""
            SELECT home_win FROM games
            WHERE home_team=? AND season=? AND week<? AND home_win IS NOT NULL
            ORDER BY game_date DESC LIMIT ?
        """, (team, season, week, n)).fetchall()
        wins = sum(1 for r in rows if r["home_win"] == 1)
    else:
        rows = conn.execute("""
            SELECT home_win FROM games
            WHERE away_team=? AND season=? AND week<? AND home_win IS NOT NULL
            ORDER BY game_date DESC LIMIT ?
        """, (team, season, week, n)).fetchall()
        wins = sum(1 for r in rows if r["home_win"] == 0)
    conn.close()
    return wins / len(rows) if rows else 0.5


def get_rest_days(db_path: str, team: str, season: int, week: int) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    prior = conn.execute("""
        SELECT game_date FROM games
        WHERE (home_team=? OR away_team=?) AND season=? AND week<?
          AND home_win IS NOT NULL
        ORDER BY game_date DESC LIMIT 1
    """, (team, team, season, week)).fetchone()
    current = conn.execute("""
        SELECT game_date FROM games
        WHERE (home_team=? OR away_team=?) AND season=? AND week=?
        LIMIT 1
    """, (team, team, season, week)).fetchone()
    conn.close()

    if not prior:
        return 14
    if not current:
        return 7
    last_date = pd.to_datetime(prior["game_date"])
    game_date = pd.to_datetime(current["game_date"])
    return max(1, (game_date - last_date).days)


def get_team_sos(db_path: str, team: str, season: int, week: int, n: int = 4) -> float:
    """Avg win% of recent opponents (strength of schedule)."""
    results = get_team_results(db_path, team, season, week, n)
    if not results:
        return 0.5
    opp_winpcts = []
    for r in results:
        opp = r["away_team"] if r["home_team"] == team else r["home_team"]
        opp_form = get_team_recent_form(db_path, opp, season, r["week"], n=4)
        opp_winpcts.append(opp_form["win_pct"])
    return sum(opp_winpcts) / len(opp_winpcts)
```

- [ ] **Step 2: Verify import works**

```bash
uv run python -c "from src.data.historical import load_games, get_team_recent_form; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/data/historical.py
git commit -m "feat: rewrite historical module as ESPN DB-backed data access layer"
```

---

## Task 6: Rewrite injuries module

**Files:**
- Rewrite: `src/data/injuries.py`

- [ ] **Step 1: Rewrite `src/data/injuries.py`**

```python
"""ESPN-backed injury fetcher. Replaces Sleeper API integration."""
from src.data.espn import fetch_team_injuries

OUT_STATUSES = {"Out", "Doubtful", "IR", "PUP-R"}


def fetch_espn_injuries(team_id: str, team_abbr: str, season: int, week: int) -> list[dict]:
    """Fetch current injury report for a team from ESPN."""
    raw = fetch_team_injuries(team_id)
    result = []
    for item in raw:
        athlete = item.get("athlete", {})
        position = athlete.get("position", {}).get("abbreviation", "")
        result.append({
            "season": season,
            "week": week,
            "team": team_abbr,
            "athlete_id": str(athlete.get("id", "")),
            "athlete_name": athlete.get("displayName", ""),
            "position": position,
            "status": item.get("status", ""),
            "is_qb": int(position == "QB"),
        })
    return result


def is_qb_out(injuries: list[dict]) -> bool:
    return any(
        i["is_qb"] and i.get("status") in OUT_STATUSES
        for i in injuries
    )
```

- [ ] **Step 2: Verify import**

```bash
uv run python -c "from src.data.injuries import fetch_espn_injuries, is_qb_out; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/data/injuries.py
git commit -m "feat: replace Sleeper injury fetcher with ESPN"
```

---

## Task 7: Update feature builder with new FEATURE_COLS

**Files:**
- Rewrite: `src/features/builder.py`

- [ ] **Step 1: Write failing feature tests**

Replace `tests/test_features.py` entirely:

```python
import pytest
from src.db.schema import create_schema
from src.db.queries import insert_espn_game, insert_team_stats, insert_injury
from src.features.builder import FEATURE_COLS, build_features_for_game, build_training_dataset


def _game(espn_id, season, week, home, away, home_score, away_score, home_win,
           game_date, is_indoor=0, game_type="regular"):
    return {
        "espn_id": espn_id, "season": season, "week": week, "game_type": game_type,
        "home_team": home, "away_team": away, "home_espn_id": "1", "away_espn_id": "2",
        "game_date": game_date, "venue": "Test Stadium",
        "is_indoor": is_indoor, "is_neutral": 0, "attendance": None,
        "home_score": home_score, "away_score": away_score, "home_win": home_win,
    }


def _stats(espn_id, team, is_home, total_yards=300, turnovers=1,
           third_made=6, third_att=14):
    return {
        "espn_id": espn_id, "team": team, "is_home": is_home,
        "total_yards": total_yards, "pass_yards": 200, "rush_yards": 100,
        "turnovers": turnovers, "first_downs": 20, "third_down_att": third_att,
        "third_down_made": third_made, "red_zone_att": 3, "red_zone_made": 2,
        "possession_secs": 1800, "sacks_taken": 2,
    }


@pytest.fixture
def db_with_history(tmp_path):
    db_path = str(tmp_path / "test.db")
    create_schema(db_path)
    # Week 1 — BAL wins
    insert_espn_game(db_path, _game("g1", 2024, 1, "BAL", "KC", 27, 20, 1, "2024-09-05T20:00Z"))
    insert_team_stats(db_path, _stats("g1", "BAL", 1, total_yards=350, turnovers=1, third_made=7, third_att=14))
    insert_team_stats(db_path, _stats("g1", "KC", 0, total_yards=280, turnovers=2, third_made=5, third_att=13))
    # Week 2 — BAL loses
    insert_espn_game(db_path, _game("g2", 2024, 2, "BAL", "LV", 14, 28, 0, "2024-09-12T20:00Z"))
    insert_team_stats(db_path, _stats("g2", "BAL", 1, total_yards=250, turnovers=3, third_made=4, third_att=14))
    insert_team_stats(db_path, _stats("g2", "LV", 0, total_yards=320, turnovers=1, third_made=8, third_att=13))
    # Week 3 upcoming (the game we're predicting features for)
    insert_espn_game(db_path, _game("g3", 2024, 3, "BAL", "PIT", None, None, None, "2024-09-19T20:00Z"))
    return db_path


def test_feature_cols_count():
    assert len(FEATURE_COLS) == 27


def test_feature_cols_has_new_espn_features():
    assert "home_turnover_diff_4wk" in FEATURE_COLS
    assert "away_turnover_diff_4wk" in FEATURE_COLS
    assert "home_total_yards_4wk" in FEATURE_COLS
    assert "away_total_yards_4wk" in FEATURE_COLS
    assert "home_third_down_pct_4wk" in FEATURE_COLS
    assert "away_third_down_pct_4wk" in FEATURE_COLS
    assert "home_qb_active" in FEATURE_COLS
    assert "away_qb_active" in FEATURE_COLS
    assert "is_indoor" in FEATURE_COLS
    assert "is_neutral" in FEATURE_COLS
    assert "home_home_winpct" in FEATURE_COLS
    assert "away_road_winpct" in FEATURE_COLS


def test_feature_cols_dropped_old_keys():
    assert "home_qb_out" not in FEATURE_COLS
    assert "away_qb_out" not in FEATURE_COLS


def test_build_features_has_all_keys(db_with_history):
    game = {
        "espn_id": "g3", "season": 2024, "week": 3,
        "game_type": "regular", "home_team": "BAL", "away_team": "PIT",
        "is_indoor": 0, "is_neutral": 0,
    }
    features = build_features_for_game(game, db_with_history, odds_home_win_prob=0.60)
    for col in FEATURE_COLS:
        assert col in features, f"Missing feature: {col}"


def test_build_features_box_stats_populated(db_with_history):
    game = {
        "espn_id": "g3", "season": 2024, "week": 3,
        "game_type": "regular", "home_team": "BAL", "away_team": "PIT",
        "is_indoor": 0, "is_neutral": 0,
    }
    features = build_features_for_game(game, db_with_history, odds_home_win_prob=0.60)
    # BAL had 350 yards in week 1 and 250 in week 2 → avg 300
    assert features["home_total_yards_4wk"] == pytest.approx(300.0)
    # BAL had 1 turnover week 1, 3 turnovers week 2 → avg -2 (negated for "fewer is better")
    assert features["home_turnover_diff_4wk"] == pytest.approx(-2.0)


def test_build_features_indoor_zeroes_weather(db_with_history):
    game = {
        "espn_id": "g3", "season": 2024, "week": 3,
        "game_type": "regular", "home_team": "BAL", "away_team": "PIT",
        "is_indoor": 1, "is_neutral": 0,
    }
    features = build_features_for_game(
        game, db_with_history, odds_home_win_prob=0.60,
        weather={"temperature": 20.0, "wind_speed": 30.0},
    )
    assert features["temperature"] == 68.0
    assert features["wind_speed"] == 0.0


def test_build_features_qb_active_defaults_to_1(db_with_history):
    game = {
        "espn_id": "g3", "season": 2024, "week": 3,
        "game_type": "regular", "home_team": "BAL", "away_team": "PIT",
        "is_indoor": 0, "is_neutral": 0,
    }
    features = build_features_for_game(game, db_with_history, odds_home_win_prob=0.60)
    assert features["home_qb_active"] == 1
    assert features["away_qb_active"] == 1


def test_build_features_qb_out_sets_inactive(db_with_history):
    insert_injury(db_with_history, {
        "season": 2024, "week": 3, "team": "BAL",
        "athlete_id": "111", "athlete_name": "Lamar Jackson",
        "position": "QB", "status": "Out", "is_qb": 1,
    })
    game = {
        "espn_id": "g3", "season": 2024, "week": 3,
        "game_type": "regular", "home_team": "BAL", "away_team": "PIT",
        "is_indoor": 0, "is_neutral": 0,
    }
    features = build_features_for_game(game, db_with_history, odds_home_win_prob=0.60)
    assert features["home_qb_active"] == 0


def test_build_training_dataset_returns_dataframe(db_with_history):
    df = build_training_dataset(db_with_history, seasons=[2024])
    assert len(df) == 2  # 2 completed games
    assert "home_win" in df.columns
    assert set(FEATURE_COLS).issubset(set(df.columns))
    assert df["home_win"].isin([0, 1]).all()
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_features.py -v
```

Expected: multiple failures — new FEATURE_COLS don't exist, `build_features_for_game` signature mismatch.

- [ ] **Step 3: Rewrite `src/features/builder.py`**

```python
import logging
import pandas as pd
from typing import Optional
from src.db.queries import get_team_box_stats, get_injuries_for_week

FEATURE_COLS = [
    "odds_home_win_prob",
    "home_rest_days", "away_rest_days", "rest_advantage",
    "home_recent_winpct", "away_recent_winpct",
    "home_home_winpct", "away_road_winpct",
    "home_recent_point_diff", "away_recent_point_diff",
    "home_turnover_diff_4wk", "away_turnover_diff_4wk",
    "home_total_yards_4wk", "away_total_yards_4wk",
    "home_third_down_pct_4wk", "away_third_down_pct_4wk",
    "home_qb_active", "away_qb_active",
    "home_key_injuries", "away_key_injuries",
    "is_indoor", "is_neutral",
    "temperature", "wind_speed",
    "home_sos", "away_sos",
    "is_playoff",
]

PLAYOFF_TYPES = {"wildcard", "divisional", "conference", "superbowl"}
OUT_STATUSES = {"Out", "Doubtful", "IR", "PUP-R"}
KEY_POSITIONS = {"QB", "RB", "WR", "TE", "OT", "OG", "C", "DE", "DT", "LB", "CB", "S"}


def _box_features(db_path: str, team: str, season: int, week: int) -> dict:
    rows = get_team_box_stats(db_path, team, season, week, n=4)
    if not rows:
        return {"turnover_diff": 0.0, "total_yards": 0.0, "third_down_pct": 0.35}
    turnover_diffs, yards, third_pcts = [], [], []
    for r in rows:
        turnover_diffs.append(-(r.get("turnovers") or 0))
        yards.append(r.get("total_yards") or 0)
        att = r.get("third_down_att") or 1
        made = r.get("third_down_made") or 0
        third_pcts.append(made / att)
    n = len(rows)
    return {
        "turnover_diff": sum(turnover_diffs) / n,
        "total_yards": sum(yards) / n,
        "third_down_pct": sum(third_pcts) / n,
    }


def _injury_features(db_path: str, team: str, season: int, week: int) -> dict:
    injuries = get_injuries_for_week(db_path, team, season, week)
    qb_out = any(
        i["is_qb"] and i.get("status") in OUT_STATUSES for i in injuries
    )
    key_out = sum(
        1 for i in injuries
        if not i["is_qb"]
        and i.get("status") in OUT_STATUSES
        and i.get("position") in KEY_POSITIONS
    )
    return {"qb_active": 0 if qb_out else 1, "key_injuries": key_out}


def build_features_for_game(
    game: dict,
    db_path: str,
    odds_home_win_prob: float,
    weather: Optional[dict] = None,
) -> dict:
    from src.data.historical import (
        get_team_recent_form, get_rest_days,
        get_home_road_winpct, get_team_sos,
    )
    season = int(game["season"])
    week = int(game["week"])
    home = game["home_team"]
    away = game["away_team"]

    home_form = get_team_recent_form(db_path, home, season, week)
    away_form = get_team_recent_form(db_path, away, season, week)
    home_rest = get_rest_days(db_path, home, season, week)
    away_rest = get_rest_days(db_path, away, season, week)
    home_home_pct = get_home_road_winpct(db_path, home, season, week, home_games=True)
    away_road_pct = get_home_road_winpct(db_path, away, season, week, home_games=False)
    home_sos = get_team_sos(db_path, home, season, week)
    away_sos = get_team_sos(db_path, away, season, week)

    home_box = _box_features(db_path, home, season, week)
    away_box = _box_features(db_path, away, season, week)
    home_inj = _injury_features(db_path, home, season, week)
    away_inj = _injury_features(db_path, away, season, week)

    is_indoor = int(game.get("is_indoor", 0))
    if is_indoor:
        temperature, wind_speed = 68.0, 0.0
    else:
        temperature = weather.get("temperature", 65.0) if weather else 65.0
        wind_speed = weather.get("wind_speed", 5.0) if weather else 5.0

    return {
        "odds_home_win_prob": odds_home_win_prob,
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "rest_advantage": home_rest - away_rest,
        "home_recent_winpct": home_form["win_pct"],
        "away_recent_winpct": away_form["win_pct"],
        "home_home_winpct": home_home_pct,
        "away_road_winpct": away_road_pct,
        "home_recent_point_diff": home_form["avg_point_diff"],
        "away_recent_point_diff": away_form["avg_point_diff"],
        "home_turnover_diff_4wk": home_box["turnover_diff"],
        "away_turnover_diff_4wk": away_box["turnover_diff"],
        "home_total_yards_4wk": home_box["total_yards"],
        "away_total_yards_4wk": away_box["total_yards"],
        "home_third_down_pct_4wk": home_box["third_down_pct"],
        "away_third_down_pct_4wk": away_box["third_down_pct"],
        "home_qb_active": home_inj["qb_active"],
        "away_qb_active": away_inj["qb_active"],
        "home_key_injuries": home_inj["key_injuries"],
        "away_key_injuries": away_inj["key_injuries"],
        "is_indoor": is_indoor,
        "is_neutral": int(game.get("is_neutral", 0)),
        "temperature": temperature,
        "wind_speed": wind_speed,
        "home_sos": home_sos,
        "away_sos": away_sos,
        "is_playoff": int(game.get("game_type", "regular") in PLAYOFF_TYPES),
    }


def build_training_dataset(
    db_path: str,
    seasons: list[int],
    odds_by_game: Optional[dict] = None,
) -> pd.DataFrame:
    from src.data.historical import load_games
    games = load_games(db_path, seasons)
    rows = []
    for _, game in games.iterrows():
        g = dict(game)
        odds_prob = (odds_by_game or {}).get(g["espn_id"], 0.55)
        try:
            features = build_features_for_game(g, db_path, odds_prob)
            features["home_win"] = int(g["home_win"])
            features["espn_id"] = g["espn_id"]
            features["season"] = int(g["season"])
            features["week"] = int(g["week"])
            features["home_team"] = g["home_team"]
            features["away_team"] = g["away_team"]
            rows.append(features)
        except Exception as e:
            logging.warning("Skipping game %s: %s", g.get("espn_id"), e)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run feature tests**

```bash
uv run pytest tests/test_features.py -v
```

Expected: all 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/features/builder.py tests/test_features.py
git commit -m "feat: new FEATURE_COLS with ESPN box score and injury features"
```

---

## Task 8: Update model, evaluate, and backtest to use espn_id

**Files:**
- Modify: `src/model/evaluate.py`
- Modify: `scripts/backtest.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Update `src/model/evaluate.py`**

Change all `game_id` references to `espn_id` and add `db_path` parameter to `run_season_backtest`:

```python
from sklearn.metrics import brier_score_loss


def baseline_accuracy(predictions: list[dict]) -> float:
    if not predictions:
        return 0.0
    return sum(1 for p in predictions if bool(p["home_win"])) / len(predictions)


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
            p.get("win_probability", p["home_win_prob"]) * p["confidence_points"]
            for p in predictions
        ),
        "brier_score": brier_score_loss(labels, probs),
        "baseline_accuracy": sum(1 for p in predictions if bool(p["home_win"])) / n,
    }


def run_season_backtest(
    model_path: str,
    seasons: list[int],
    db_path: str,
    point_range: tuple = (1, 16),
) -> list[dict]:
    from src.features.builder import build_training_dataset, FEATURE_COLS
    from src.model.predict import predict_week
    from src.optimizer.confidence import assign_confidence_points

    all_metrics = []
    for season in seasons:
        games_df = build_training_dataset(db_path=db_path, seasons=[season])
        weeks = sorted(games_df["week"].unique())
        for week in weeks:
            week_games = games_df[games_df["week"] == week]
            game_inputs = []
            for _, row in week_games.iterrows():
                game_inputs.append({
                    "espn_id": row["espn_id"],
                    "home_team": row.get("home_team", ""),
                    "away_team": row.get("away_team", ""),
                    "features": {col: row[col] for col in FEATURE_COLS if col in row.index},
                })
            if not game_inputs:
                continue
            predictions = predict_week(model_path, game_inputs)
            assignments = assign_confidence_points(predictions, point_range)
            assign_by_id = {a["espn_id"]: a["confidence_points"] for a in assignments}
            for pred in predictions:
                pred["confidence_points"] = assign_by_id[pred["espn_id"]]
                eid = pred["espn_id"]
                home_win_rows = week_games[week_games["espn_id"] == eid]["home_win"]
                if len(home_win_rows) == 0:
                    raise ValueError(f"espn_id {eid} not found in week_games")
                pred["home_win"] = int(home_win_rows.values[0])
            metrics = compute_week_metrics(predictions)
            metrics["season"] = season
            metrics["week"] = week
            all_metrics.append(metrics)
    return all_metrics
```

- [ ] **Step 2: Update `src/model/predict.py`**

Change `game["game_id"]` to `game["espn_id"]`:

```python
import pandas as pd
from src.model.train import load_model
from src.features.builder import FEATURE_COLS


def predict_game_prob(model_path: str, features: dict) -> float:
    artifact = load_model(model_path)
    model = artifact["model"]
    medians = artifact.get("medians", {})
    X = pd.DataFrame([features])[FEATURE_COLS].fillna(medians).fillna(0.5)
    return float(model.predict_proba(X)[0][1])


def predict_week(model_path: str, games_features: list[dict]) -> list[dict]:
    artifact = load_model(model_path)
    model = artifact["model"]
    medians = artifact.get("medians", {})
    results = []
    for game in games_features:
        X = pd.DataFrame([game["features"]])[FEATURE_COLS].fillna(medians).fillna(0.5)
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

- [ ] **Step 3: Update `scripts/backtest.py`**

```python
#!/usr/bin/env python3
"""Backtest the model on validation seasons and print summary."""
import yaml
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
    train_df = build_training_dataset(db_path=db_path, seasons=train_seasons)
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

- [ ] **Step 4: Update `tests/test_model.py`**

Replace `"home_qb_out"` and `"away_qb_out"` references with `"home_qb_active"` and `"away_qb_active"`, and fix `game_id` → `espn_id` in `predict_week` test:

```python
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.model.train import train_model, load_model
from src.model.predict import predict_game_prob, predict_week
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
    assert load_model(model_path) is not None


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
    base["home_qb_active"] = 1
    base["away_qb_active"] = 1
    high_odds = {**base, "odds_home_win_prob": 0.80}
    low_odds = {**base, "odds_home_win_prob": 0.40}
    assert predict_game_prob(model_path, high_odds) > predict_game_prob(model_path, low_odds)


def test_predict_week_returns_required_keys(tmp_path):
    df = make_fake_training_data()
    model_path = str(tmp_path / "model.joblib")
    train_model(df, model_path)
    games = [{
        "espn_id": "401220225",
        "home_team": "KC",
        "away_team": "BAL",
        "features": {col: 0.5 for col in FEATURE_COLS},
    }]
    results = predict_week(model_path, games)
    assert len(results) == 1
    r = results[0]
    assert "home_win_prob" in r
    assert "predicted_winner" in r
    assert "win_probability" in r
    assert 0.0 <= r["home_win_prob"] <= 1.0
    assert r["predicted_winner"] in ("KC", "BAL")


from src.model.evaluate import compute_week_metrics, baseline_accuracy


def test_baseline_accuracy_always_picks_favorite():
    predictions = [
        {"home_win_prob": 0.70, "home_win": 1},
        {"home_win_prob": 0.40, "home_win": 1},
        {"home_win_prob": 0.65, "home_win": 0},
    ]
    acc = baseline_accuracy(predictions)
    assert abs(acc - 2 / 3) < 0.01


def test_compute_week_metrics():
    predictions = [
        {"home_win_prob": 0.70, "home_win": 1, "confidence_points": 3},
        {"home_win_prob": 0.60, "home_win": 0, "confidence_points": 2},
        {"home_win_prob": 0.55, "home_win": 1, "confidence_points": 1},
    ]
    metrics = compute_week_metrics(predictions)
    assert metrics["actual_points"] == 4
    assert metrics["accuracy"] == pytest.approx(2 / 3)
```

- [ ] **Step 5: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass (excluding `test_db.py` tests that reference old `insert_game` or Sleeper — those will be cleaned up next).

- [ ] **Step 6: Commit**

```bash
git add src/model/evaluate.py src/model/predict.py scripts/backtest.py tests/test_model.py
git commit -m "feat: update model/evaluate/backtest to use espn_id and db_path"
```

---

## Task 9: Clean up remaining stale tests and finalize test_db.py

**Files:**
- Rewrite: `tests/test_db.py` (remove stale Sleeper/insert_game tests, add ESPN tests)

- [ ] **Step 1: Replace full `tests/test_db.py`**

Keep only tests that work with the new schema. The Sleeper and old `insert_game` tests are removed; the odds and weather tests stay since those modules are unchanged.

```python
import sqlite3
import os
import pytest
from src.db.schema import create_schema


def test_schema_creates_all_tables():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        create_schema(db_path)
        conn = sqlite3.connect(db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        expected = {
            "games", "team_game_stats", "injury_reports", "depth_charts",
            "predictions", "weekly_assignments", "conversations",
            "rerankings", "model_metrics", "family_picks",
        }
        assert expected.issubset(tables)
    finally:
        os.unlink(db_path)


@pytest.fixture
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    create_schema(db_path)
    return db_path


def test_insert_and_fetch_espn_game(tmp_db):
    from src.db.queries import insert_espn_game, get_games_for_week
    game = {
        "espn_id": "401220225", "season": 2024, "week": 1, "game_type": "regular",
        "home_team": "BAL", "away_team": "KC", "home_espn_id": "33",
        "away_espn_id": "12", "game_date": "2024-09-05T20:00Z",
        "venue": "M&T Bank Stadium", "is_indoor": 0, "is_neutral": 0,
        "attendance": 71000, "home_score": 27, "away_score": 20, "home_win": 1,
    }
    insert_espn_game(tmp_db, game)
    games = get_games_for_week(tmp_db, 2024, 1)
    assert len(games) == 1
    assert games[0]["espn_id"] == "401220225"
    assert games[0]["home_win"] == 1


def test_get_existing_espn_ids(tmp_db):
    from src.db.queries import insert_espn_game, get_existing_espn_ids
    insert_espn_game(tmp_db, {
        "espn_id": "999", "season": 2024, "week": 1, "game_type": "regular",
        "home_team": "BAL", "away_team": "KC", "home_espn_id": "33",
        "away_espn_id": "12", "game_date": "2024-09-05T20:00Z",
        "venue": "X", "is_indoor": 0, "is_neutral": 0,
        "attendance": None, "home_score": None, "away_score": None, "home_win": None,
    })
    assert "999" in get_existing_espn_ids(tmp_db)


def test_insert_and_fetch_team_stats(tmp_db):
    from src.db.queries import insert_espn_game, insert_team_stats, get_team_box_stats
    insert_espn_game(tmp_db, {
        "espn_id": "401220225", "season": 2024, "week": 1, "game_type": "regular",
        "home_team": "BAL", "away_team": "KC", "home_espn_id": "33",
        "away_espn_id": "12", "game_date": "2024-09-05T20:00Z", "venue": "M",
        "is_indoor": 0, "is_neutral": 0, "attendance": 71000,
        "home_score": 27, "away_score": 20, "home_win": 1,
    })
    insert_team_stats(tmp_db, {
        "espn_id": "401220225", "team": "BAL", "is_home": 1,
        "total_yards": 350, "pass_yards": 220, "rush_yards": 130,
        "turnovers": 1, "first_downs": 22, "third_down_att": 14,
        "third_down_made": 7, "red_zone_att": 4, "red_zone_made": 3,
        "possession_secs": 1920, "sacks_taken": 1,
    })
    rows = get_team_box_stats(tmp_db, "BAL", 2024, 2)
    assert len(rows) == 1
    assert rows[0]["total_yards"] == 350


def test_get_team_results_ordering(tmp_db):
    from src.db.queries import insert_espn_game, get_team_results
    for espn_id, week, hw in [("g1", 1, 1), ("g2", 2, 0), ("g3", 3, 1)]:
        insert_espn_game(tmp_db, {
            "espn_id": espn_id, "season": 2024, "week": week, "game_type": "regular",
            "home_team": "BAL", "away_team": "KC", "home_espn_id": "33",
            "away_espn_id": "12", "game_date": f"2024-09-0{week + 4}T20:00Z",
            "venue": "M", "is_indoor": 0, "is_neutral": 0, "attendance": None,
            "home_score": 27 if hw else 14, "away_score": 20 if hw else 28, "home_win": hw,
        })
    results = get_team_results(tmp_db, "BAL", 2024, before_week=4, n=4)
    assert len(results) == 3
    assert results[0]["espn_id"] == "g3"  # most recent first


def test_insert_and_fetch_injury(tmp_db):
    from src.db.queries import insert_injury, get_injuries_for_week
    insert_injury(tmp_db, {
        "season": 2024, "week": 5, "team": "KC",
        "athlete_id": "3139477", "athlete_name": "Patrick Mahomes",
        "position": "QB", "status": "Questionable", "is_qb": 1,
    })
    rows = get_injuries_for_week(tmp_db, "KC", 2024, 5)
    assert len(rows) == 1
    assert rows[0]["athlete_name"] == "Patrick Mahomes"


def test_insert_and_fetch_depth_chart(tmp_db):
    from src.db.queries import insert_depth_chart_entry, get_starting_qb
    insert_depth_chart_entry(tmp_db, {
        "season": 2024, "week": 5, "team": "KC",
        "athlete_id": "3139477", "athlete_name": "Patrick Mahomes", "rank": 1,
    })
    qb = get_starting_qb(tmp_db, "KC", 2024, 5)
    assert qb is not None
    assert qb["athlete_name"] == "Patrick Mahomes"


from src.data.odds import moneyline_to_prob, parse_odds_response


def test_moneyline_to_prob_negative_favorite():
    assert abs(moneyline_to_prob(-200) - 0.667) < 0.01


def test_moneyline_to_prob_positive_underdog():
    assert abs(moneyline_to_prob(150) - 0.400) < 0.01


def test_parse_odds_response_removes_vig():
    fake_response = [{
        "id": "abc123",
        "home_team": "Kansas City Chiefs",
        "away_team": "Baltimore Ravens",
        "bookmakers": [{"markets": [{"key": "h2h", "outcomes": [
            {"name": "Kansas City Chiefs", "price": -180},
            {"name": "Baltimore Ravens", "price": 155},
        ]}]}],
    }]
    result = parse_odds_response(fake_response)
    assert len(result) == 1
    total = result[0]["home_win_prob"] + result[0]["away_win_prob"]
    assert abs(total - 1.0) < 0.001


from src.data.weather import estimate_weather_impact


def test_weather_impact_indoor_is_zero():
    assert estimate_weather_impact(is_outdoor=False, temperature=32, wind_speed=20) == 0.0


def test_weather_impact_cold_wind_negative():
    assert estimate_weather_impact(is_outdoor=True, temperature=20, wind_speed=25) < 0
```

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_db.py
git commit -m "test: finalize test_db.py for ESPN schema"
```

---

## Task 10: Rewrite historical backfill script

**Files:**
- Rewrite: `scripts/ingest_historical.py`

- [ ] **Step 1: Rewrite `scripts/ingest_historical.py`**

```python
#!/usr/bin/env python3
"""
Backfill ESPN game data for 2018-2025 into SQLite.
Checkpoints on espn_id: already-ingested games are skipped.
Rate: 1 req/sec. Estimated time: ~50 minutes for full backfill.
"""
import time
import yaml
from src.db.schema import create_schema
from src.db.queries import (
    get_existing_espn_ids, insert_espn_game, insert_team_stats, insert_injury,
)
from src.data.espn import (
    fetch_scoreboard, fetch_game_summary, parse_game,
    parse_box_score, parse_game_injuries,
)

SEASONS = list(range(2018, 2026))
# (season_type, week_range, label)
WEEK_CONFIGS = [
    (2, range(1, 23), "regular"),   # regular season; ESPN returns empty for non-existent weeks
    (3, range(1, 6), "postseason"),
]


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    db_path = config["db"]["path"]

    create_schema(db_path)
    existing = get_existing_espn_ids(db_path)
    print(f"Checkpoint: {len(existing)} games already in DB.")

    total_new = 0
    for season in SEASONS:
        for season_type, weeks, label in WEEK_CONFIGS:
            for week in weeks:
                try:
                    events = fetch_scoreboard(season, week, season_type)
                except Exception as e:
                    print(f"  Scoreboard error {season} {label} wk{week}: {e}")
                    time.sleep(2)
                    continue

                if not events:
                    continue

                for event in events:
                    espn_id = str(event["id"])
                    if espn_id in existing:
                        continue

                    game = parse_game(event)
                    if game["home_win"] is None:
                        # Skip incomplete/upcoming games during historical backfill
                        continue

                    insert_espn_game(db_path, game)

                    try:
                        time.sleep(1.0)
                        summary = fetch_game_summary(espn_id)
                        for stats in parse_box_score(summary, espn_id):
                            if stats["team"]:
                                insert_team_stats(db_path, stats)
                        for injury in parse_game_injuries(
                            summary, game["season"], game["week"]
                        ):
                            if injury["athlete_id"]:
                                insert_injury(db_path, injury)
                    except Exception as e:
                        print(f"  Warning: summary failed for {espn_id}: {e}")

                    existing.add(espn_id)
                    total_new += 1

                print(
                    f"  {season} {label} week {week}: "
                    f"+{len([e for e in events if str(e['id']) in existing])} games"
                )

    print(f"\nIngestion complete. Added {total_new} new games.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses without error (dry run)**

```bash
uv run python -c "
import scripts.ingest_historical as s
print('Script imports OK')
print('SEASONS:', s.SEASONS)
"
```

Expected: `Script imports OK` and `SEASONS: [2018, 2019, ..., 2025]`

- [ ] **Step 3: Commit**

```bash
git add scripts/ingest_historical.py
git commit -m "feat: rewrite backfill script with ESPN API and espn_id checkpointing"
```

---

## Task 11: Final verification

- [ ] **Step 1: Run full test suite clean**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: all tests pass. Note which tests are skipped or xfail if any.

- [ ] **Step 2: Verify feature count**

```bash
uv run python -c "
from src.features.builder import FEATURE_COLS
print(f'Feature count: {len(FEATURE_COLS)}')
print(FEATURE_COLS)
"
```

Expected: `Feature count: 27` and the list printed cleanly.

- [ ] **Step 3: Verify nfl_data_py is gone**

```bash
grep -r "nfl_data_py\|import nfl\|sleeper" src/ scripts/ tests/ --include="*.py"
```

Expected: no matches.

- [ ] **Step 4: Verify espn module imports cleanly**

```bash
uv run python -c "
from src.data.espn import fetch_scoreboard, parse_game, parse_box_score
from src.data.historical import load_games, get_team_recent_form
from src.data.injuries import fetch_espn_injuries, is_qb_out
from src.features.builder import build_features_for_game, build_training_dataset
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 5: Final commit and push**

```bash
git add -A
git commit -m "chore: final cleanup and verification"
git push origin HEAD
```

Then open a PR targeting `main` with title: `feat: replace nfl_data_py with ESPN API data pipeline`.

---

## Known Limitations

- **Historical injury/depth chart data:** The `parse_game_injuries` function extracts injury data embedded in the ESPN game summary. For seasons where this field is absent, injury features default to `home_qb_active=1` / `home_key_injuries=0`. Real injury history (pre-game, week-by-week) is only available going forward via `teams/{id}/injuries`.

- **Box score stats availability:** ESPN box score statistics are populated in the `summary` endpoint for completed games. If a game's summary lacks a `boxscore.teams` array (rare for older seasons), box score features default to 0 / 0.35 (third-down rate median).

- **ESPN API reliability:** The ESPN unofficial API has no SLA. The backfill script adds `time.sleep(1.0)` between summary fetches and catches per-game exceptions so a single failure doesn't abort the full run.
