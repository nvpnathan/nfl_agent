# ESPN Odds Integration Implementation Plan

> **Status:** Historical implementation plan snapshot from 2026-05-09.
> It is preserved for planning history and may reference migration details that are already completed or superseded.
> Use `README.md` and `docs/model-card.md` for current behavior.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fake `odds_home_win_prob=0.55` training constant with real Bet365 pre-match spread and totals from ESPN's core API, switch calibration from isotonic to Platt (sigmoid), and add walk-forward validation to the backtest.

**Architecture:** Store historical Bet365 odds (spread + total) in a new `game_odds` table. The feature builder drops `odds_home_win_prob` and adds `home_spread` + `game_total` as the market anchor. The model learns situational corrections on top of the spread signal. The backtest uses rolling walk-forward folds (train 2019–N, validate N+1) instead of random CV so accuracy estimates reflect real prediction scenarios.

**Tech Stack:** SQLite, ESPN core API (`sports.core.api.espn.com/v2`), XGBoost, sklearn `CalibratedClassifierCV(method="sigmoid")`, pytest, uv

---

## File Map

| File | Change |
|---|---|
| `src/db/schema.py` | Add `game_odds` table |
| `src/db/queries.py` | Add `insert_game_odds`, `get_game_odds` |
| `src/data/espn.py` | Add `fetch_game_odds(espn_id)` |
| `scripts/backfill_odds.py` | New — backfill Bet365 odds for all games 2019–2024 |
| `src/features/builder.py` | Replace `odds_home_win_prob` → `home_spread` + `game_total` in `FEATURE_COLS` and `build_features_for_game` |
| `src/model/train.py` | Change `method="isotonic"` → `method="sigmoid"` |
| `scripts/backtest.py` | Replace static train/val split with walk-forward folds |
| `tests/test_db.py` | Add odds insert/get tests |
| `tests/test_odds.py` | New — unit tests for `fetch_game_odds` parsing |
| `tests/test_features.py` | Update for new signature + feature count |
| `tests/test_model.py` | Update fake data generator + predictor test |

---

### Task 1: Add `game_odds` table + query functions

**Files:**
- Modify: `src/db/schema.py`
- Modify: `src/db/queries.py`
- Modify: `tests/test_db.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_db.py`:

```python
from src.db.queries import insert_game_odds, get_game_odds

def test_insert_and_get_game_odds(tmp_path):
    db_path = str(tmp_path / "test.db")
    create_schema(db_path)
    insert_espn_game(db_path, _game("g1", 2024, 1, "KC", "BAL", 27, 20, 1, "2024-09-05T20:00Z"))
    insert_game_odds(db_path, {
        "espn_id": "g1",
        "home_spread": -2.5,
        "game_total": 46.5,
        "home_moneyline": "4/6",
        "away_moneyline": "5/4",
    })
    row = get_game_odds(db_path, "g1")
    assert row is not None
    assert row["home_spread"] == -2.5
    assert row["game_total"] == 46.5
    assert row["home_moneyline"] == "4/6"

def test_get_game_odds_returns_none_for_missing(tmp_path):
    db_path = str(tmp_path / "test.db")
    create_schema(db_path)
    assert get_game_odds(db_path, "nonexistent") is None

def test_insert_game_odds_upserts(tmp_path):
    db_path = str(tmp_path / "test.db")
    create_schema(db_path)
    insert_espn_game(db_path, _game("g1", 2024, 1, "KC", "BAL", 27, 20, 1, "2024-09-05T20:00Z"))
    insert_game_odds(db_path, {"espn_id": "g1", "home_spread": -2.5, "game_total": 46.5,
                               "home_moneyline": "4/6", "away_moneyline": "5/4"})
    insert_game_odds(db_path, {"espn_id": "g1", "home_spread": -3.0, "game_total": 47.0,
                               "home_moneyline": "1/2", "away_moneyline": "7/4"})
    row = get_game_odds(db_path, "g1")
    assert row["home_spread"] == -3.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_db.py::test_insert_and_get_game_odds tests/test_db.py::test_get_game_odds_returns_none_for_missing tests/test_db.py::test_insert_game_odds_upserts -v
```

Expected: FAIL with `ImportError: cannot import name 'insert_game_odds'`

- [ ] **Step 3: Add `game_odds` table to schema**

In `src/db/schema.py`, add after the `depth_charts` table block (before `predictions`):

```sql
        CREATE TABLE IF NOT EXISTS game_odds (
            espn_id TEXT PRIMARY KEY,
            home_spread REAL NOT NULL,
            game_total REAL NOT NULL,
            home_moneyline TEXT,
            away_moneyline TEXT,
            fetched_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (espn_id) REFERENCES games(espn_id)
        );
```

- [ ] **Step 4: Add `insert_game_odds` and `get_game_odds` to queries**

In `src/db/queries.py`, add after `insert_depth_chart_entry`:

```python
def insert_game_odds(db_path: str, odds: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO game_odds
            (espn_id, home_spread, game_total, home_moneyline, away_moneyline)
            VALUES (:espn_id, :home_spread, :game_total, :home_moneyline, :away_moneyline)
        """, odds)


def get_game_odds(db_path: str, espn_id: str) -> dict | None:
    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM game_odds WHERE espn_id = ?", (espn_id,)
        ).fetchone()
    return dict(row) if row else None
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
uv run pytest tests/test_db.py::test_insert_and_get_game_odds tests/test_db.py::test_get_game_odds_returns_none_for_missing tests/test_db.py::test_insert_game_odds_upserts -v
```

Expected: PASS (3 tests)

- [ ] **Step 6: Run full test suite to check for regressions**

```bash
uv run pytest tests/ -v
```

Expected: all existing tests pass

- [ ] **Step 7: Commit**

```bash
git add src/db/schema.py src/db/queries.py tests/test_db.py
git commit -m "feat: add game_odds table and insert/get query functions"
```

---

### Task 2: ESPN odds fetcher + backfill script

**Files:**
- Modify: `src/data/espn.py`
- Create: `scripts/backfill_odds.py`
- Create: `tests/test_odds.py`

- [ ] **Step 1: Write failing unit tests for the fetcher parser**

Create `tests/test_odds.py`:

```python
from unittest.mock import patch, MagicMock
from src.data.espn import fetch_game_odds


def _make_espn_response(spread, total, home_ml, away_ml):
    return {
        "items": [{
            "provider": {"id": "2000", "name": "Bet 365"},
            "bettingOdds": {
                "teamOdds": {
                    "preMatchSpreadHandicapHome": {"value": spread},
                    "preMatchTotalHandicap": {"value": total},
                    "preMatchMoneyLineHome": {"value": home_ml},
                    "preMatchMoneyLineAway": {"value": away_ml},
                }
            }
        }]
    }


def test_fetch_game_odds_parses_bet365_fields():
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = _make_espn_response("-2.5", "46.5", "4/6", "5/4")
    with patch("src.data.espn.requests.get", return_value=mock_resp):
        result = fetch_game_odds("401671789")
    assert result is not None
    assert result["espn_id"] == "401671789"
    assert result["home_spread"] == -2.5
    assert result["game_total"] == 46.5
    assert result["home_moneyline"] == "4/6"
    assert result["away_moneyline"] == "5/4"


def test_fetch_game_odds_returns_none_when_no_bet365():
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {
        "items": [{"provider": {"id": "58", "name": "ESPN BET"}, "bettingOdds": {}}]
    }
    with patch("src.data.espn.requests.get", return_value=mock_resp):
        result = fetch_game_odds("401671789")
    assert result is None


def test_fetch_game_odds_returns_none_when_spread_missing():
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {
        "items": [{
            "provider": {"id": "2000", "name": "Bet 365"},
            "bettingOdds": {
                "teamOdds": {
                    "preMatchSpreadHandicapHome": {"value": None},
                    "preMatchTotalHandicap": {"value": "46.5"},
                    "preMatchMoneyLineHome": {"value": "4/6"},
                    "preMatchMoneyLineAway": {"value": "5/4"},
                }
            }
        }]
    }
    with patch("src.data.espn.requests.get", return_value=mock_resp):
        result = fetch_game_odds("401671789")
    assert result is None


def test_fetch_game_odds_returns_none_on_http_error():
    mock_resp = MagicMock()
    mock_resp.ok = False
    with patch("src.data.espn.requests.get", return_value=mock_resp):
        result = fetch_game_odds("401671789")
    assert result is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_odds.py -v
```

Expected: FAIL with `ImportError: cannot import name 'fetch_game_odds'`

- [ ] **Step 3: Add `fetch_game_odds` to `src/data/espn.py`**

Add at the end of `src/data/espn.py`:

```python
def fetch_game_odds(espn_id: str) -> dict | None:
    """Fetch Bet365 pre-match spread and total from ESPN. Returns None if unavailable."""
    url = (
        f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
        f"/events/{espn_id}/competitions/{espn_id}/odds"
    )
    resp = requests.get(url, timeout=10)
    if not resp.ok:
        return None
    data = resp.json()
    bet365 = next(
        (i for i in data.get("items", []) if i.get("provider", {}).get("id") == "2000"),
        None,
    )
    if not bet365:
        return None
    team_odds = bet365.get("bettingOdds", {}).get("teamOdds", {})
    spread = team_odds.get("preMatchSpreadHandicapHome", {}).get("value")
    total = team_odds.get("preMatchTotalHandicap", {}).get("value")
    if spread is None or total is None:
        return None
    return {
        "espn_id": espn_id,
        "home_spread": float(spread),
        "game_total": float(total),
        "home_moneyline": team_odds.get("preMatchMoneyLineHome", {}).get("value"),
        "away_moneyline": team_odds.get("preMatchMoneyLineAway", {}).get("value"),
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_odds.py -v
```

Expected: PASS (4 tests)

- [ ] **Step 5: Create `scripts/backfill_odds.py`**

```python
#!/usr/bin/env python3
"""Backfill Bet365 pre-match odds from ESPN for all games with season >= 2019.

Run after ingest_historical.py. Safe to re-run — skips games already in game_odds.
Rate-limited to 2 req/sec. Expect ~20 minutes for a full backfill of 2019–2024.
"""
import sqlite3
import time
import yaml
from src.db.schema import create_schema
from src.db.queries import insert_game_odds, get_game_odds
from src.data.espn import fetch_game_odds


def main() -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    db_path = config["db"]["path"]
    create_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT espn_id FROM games WHERE season >= 2019 AND home_win IS NOT NULL ORDER BY season, week"
        ).fetchall()
    espn_ids = [r[0] for r in rows]
    print(f"Found {len(espn_ids)} completed games from 2019+")

    already, fetched, missing = 0, 0, 0
    for espn_id in espn_ids:
        if get_game_odds(db_path, espn_id):
            already += 1
            continue
        odds = fetch_game_odds(espn_id)
        if odds:
            insert_game_odds(db_path, odds)
            fetched += 1
        else:
            missing += 1
        time.sleep(0.5)

    print(f"Done. already={already} fetched={fetched} no_data={missing}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add src/data/espn.py scripts/backfill_odds.py tests/test_odds.py
git commit -m "feat: add fetch_game_odds and backfill_odds script"
```

---

### Task 3: Update feature builder

**Files:**
- Modify: `src/features/builder.py`
- Modify: `tests/test_features.py`

- [ ] **Step 1: Update the failing tests**

In `tests/test_features.py`, make these changes:

Replace `test_feature_cols_count`:
```python
def test_feature_cols_count():
    assert len(FEATURE_COLS) == 28
```

Replace the call signature in `test_build_features_has_all_keys`:
```python
def test_build_features_has_all_keys(db_with_history):
    game = {
        "espn_id": "g3", "season": 2024, "week": 3,
        "game_type": "regular", "home_team": "BAL", "away_team": "PIT",
        "is_indoor": 0, "is_neutral": 0,
    }
    features = build_features_for_game(game, db_with_history, home_spread=-3.0, game_total=45.5)
    for col in FEATURE_COLS:
        assert col in features, f"Missing feature: {col}"
```

Replace the call signature in `test_build_features_box_stats_populated`:
```python
def test_build_features_box_stats_populated(db_with_history):
    game = {
        "espn_id": "g3", "season": 2024, "week": 3,
        "game_type": "regular", "home_team": "BAL", "away_team": "PIT",
        "is_indoor": 0, "is_neutral": 0,
    }
    features = build_features_for_game(game, db_with_history, home_spread=-3.0, game_total=45.5)
    assert features["home_total_yards_4wk"] == pytest.approx(300.0)
    assert features["home_turnover_diff_4wk"] == pytest.approx(-2.0)
```

Replace the call signature in `test_build_features_indoor_zeroes_weather`:
```python
def test_build_features_indoor_zeroes_weather(db_with_history):
    game = {
        "espn_id": "g3", "season": 2024, "week": 3,
        "game_type": "regular", "home_team": "BAL", "away_team": "PIT",
        "is_indoor": 1, "is_neutral": 0,
    }
    features = build_features_for_game(
        game, db_with_history, home_spread=-3.0, game_total=45.5,
        weather={"temperature": 20.0, "wind_speed": 30.0},
    )
    assert features["temperature"] == 68.0
    assert features["wind_speed"] == 0.0
```

Replace the call signature in `test_build_features_qb_active_defaults_to_1`:
```python
def test_build_features_qb_active_defaults_to_1(db_with_history):
    game = {
        "espn_id": "g3", "season": 2024, "week": 3,
        "game_type": "regular", "home_team": "BAL", "away_team": "PIT",
        "is_indoor": 0, "is_neutral": 0,
    }
    features = build_features_for_game(game, db_with_history, home_spread=-3.0, game_total=45.5)
    assert features["home_qb_active"] == 1
    assert features["away_qb_active"] == 1
```

Replace the call signature in `test_build_features_qb_out_sets_inactive`:
```python
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
    features = build_features_for_game(game, db_with_history, home_spread=-3.0, game_total=45.5)
    assert features["home_qb_active"] == 0
```

Add a new test for the new feature values, and update `test_build_training_dataset_returns_dataframe` to insert odds into the DB:

```python
def test_build_features_spread_and_total_in_output(db_with_history):
    game = {
        "espn_id": "g3", "season": 2024, "week": 3,
        "game_type": "regular", "home_team": "BAL", "away_team": "PIT",
        "is_indoor": 0, "is_neutral": 0,
    }
    features = build_features_for_game(game, db_with_history, home_spread=-6.5, game_total=44.0)
    assert features["home_spread"] == -6.5
    assert features["game_total"] == 44.0


def test_build_training_dataset_returns_dataframe(db_with_history):
    from src.db.queries import insert_game_odds
    insert_game_odds(db_with_history, {"espn_id": "g1", "home_spread": -3.0, "game_total": 45.5,
                                        "home_moneyline": "4/6", "away_moneyline": "5/4"})
    insert_game_odds(db_with_history, {"espn_id": "g2", "home_spread": 2.5, "game_total": 47.0,
                                        "home_moneyline": "5/4", "away_moneyline": "4/6"})
    df = build_training_dataset(db_with_history, seasons=[2024])
    assert len(df) == 2  # 2 completed games with odds
    assert "home_win" in df.columns
    assert set(FEATURE_COLS).issubset(set(df.columns))
    assert df["home_win"].isin([0, 1]).all()
```

Also update `test_feature_cols_has_new_espn_features` to check for the new columns and remove `odds_home_win_prob`:

```python
def test_feature_cols_has_new_espn_features():
    assert "home_spread" in FEATURE_COLS
    assert "game_total" in FEATURE_COLS
    assert "odds_home_win_prob" not in FEATURE_COLS
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_features.py -v
```

Expected: multiple FAILs (wrong count, missing home_spread/game_total, wrong signature)

- [ ] **Step 3: Update `src/features/builder.py`**

Replace the `FEATURE_COLS` list:

```python
FEATURE_COLS = [
    "home_spread", "game_total",
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
```

Replace the `build_features_for_game` signature and its return dict:

```python
def build_features_for_game(
    game: dict,
    db_path: str,
    home_spread: float,
    game_total: float,
    weather: Optional[dict] = None,
) -> dict:
```

In the return dict of `build_features_for_game`, replace:
```python
    return {
        "odds_home_win_prob": odds_home_win_prob,
        ...
    }
```
with:
```python
    return {
        "home_spread": home_spread,
        "game_total": game_total,
        ...
    }
```

Replace `build_training_dataset` entirely:

```python
def build_training_dataset(
    db_path: str,
    seasons: list[int],
) -> pd.DataFrame:
    from src.data.historical import load_games
    from src.db.queries import get_game_odds
    games = load_games(db_path, seasons)
    rows = []
    for _, game in games.iterrows():
        g = dict(game)
        odds = get_game_odds(db_path, g["espn_id"])
        if odds is None:
            continue  # exclude games without odds — no imputation in training
        try:
            features = build_features_for_game(
                g, db_path,
                home_spread=odds["home_spread"],
                game_total=odds["game_total"],
            )
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

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_features.py -v
```

Expected: PASS (all tests)

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: test_model.py will now fail (it uses `odds_home_win_prob` in fake data) — that is expected and will be fixed in Task 4.

- [ ] **Step 6: Commit**

```bash
git add src/features/builder.py tests/test_features.py
git commit -m "feat: replace odds_home_win_prob with home_spread + game_total in feature builder"
```

---

### Task 4: Update model training + fix model tests

**Files:**
- Modify: `src/model/train.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Update `tests/test_model.py`**

Replace `make_fake_training_data`:

```python
def make_fake_training_data(n=200) -> pd.DataFrame:
    np.random.seed(42)
    df = pd.DataFrame({col: np.random.random(n) for col in FEATURE_COLS})
    # Spread ranges from -14 to +14; negative = home favorite
    df["home_spread"] = np.random.uniform(-14, 14, n)
    df["game_total"] = np.random.uniform(38, 58, n)
    # Home team wins more often when spread is negative (home favored)
    df["home_win"] = (-df["home_spread"] + np.random.normal(0, 3, n) > 0).astype(int)
    return df
```

Replace `test_predict_game_prob_returns_valid_probability`:

```python
def test_predict_game_prob_returns_valid_probability(tmp_path):
    df = make_fake_training_data()
    model_path = str(tmp_path / "model.joblib")
    train_model(df, model_path)
    features = {col: 0.5 for col in FEATURE_COLS}
    features["home_spread"] = -3.0
    features["game_total"] = 45.5
    prob = predict_game_prob(model_path, features)
    assert 0.0 <= prob <= 1.0
```

Replace `test_predict_favors_higher_odds_team`:

```python
def test_predict_favors_spread_favorite(tmp_path):
    df = make_fake_training_data(n=500)
    model_path = str(tmp_path / "model.joblib")
    train_model(df, model_path)
    base = {col: 0.5 for col in FEATURE_COLS}
    base["home_qb_active"] = 1
    base["away_qb_active"] = 1
    base["game_total"] = 45.5
    heavy_favorite = {**base, "home_spread": -10.0}
    heavy_underdog = {**base, "home_spread": 10.0}
    assert predict_game_prob(model_path, heavy_favorite) > predict_game_prob(model_path, heavy_underdog)
```

Replace `test_predict_week_returns_required_keys`:

```python
def test_predict_week_returns_required_keys(tmp_path):
    df = make_fake_training_data()
    model_path = str(tmp_path / "model.joblib")
    train_model(df, model_path)
    features = {col: 0.5 for col in FEATURE_COLS}
    features["home_spread"] = -3.0
    features["game_total"] = 45.5
    games = [{
        "espn_id": "401220225",
        "home_team": "KC",
        "away_team": "BAL",
        "features": features,
    }]
    results = predict_week(model_path, games)
    assert len(results) == 1
    r = results[0]
    assert "home_win_prob" in r
    assert "predicted_winner" in r
    assert "win_probability" in r
    assert 0.0 <= r["home_win_prob"] <= 1.0
    assert r["predicted_winner"] in ("KC", "BAL")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_model.py -v
```

Expected: FAIL (old feature columns in fake data)

- [ ] **Step 3: Switch calibration to Platt in `src/model/train.py`**

Change line:
```python
    model = CalibratedClassifierCV(base, cv=5, method="isotonic")
```
to:
```python
    model = CalibratedClassifierCV(base, cv=5, method="sigmoid")
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
uv run pytest tests/test_model.py -v
```

Expected: PASS (all tests)

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/model/train.py tests/test_model.py
git commit -m "feat: switch calibration to Platt sigmoid, update model tests for spread features"
```

---

### Task 5: Walk-forward backtest

**Files:**
- Modify: `scripts/backtest.py`

No new tests needed — the backtest script is an executable, not a library. Verify it runs end-to-end.

- [ ] **Step 1: Replace `scripts/backtest.py` entirely**

```python
#!/usr/bin/env python3
"""Walk-forward backtest: train on 2019–N, validate on N+1 for N in [2021, 2022, 2023].

Reports accuracy, Brier score, expected confidence points, and actual confidence points
per fold and averaged across folds. Expected points and actual points are the key metrics
for the confidence pool goal.
"""
import yaml
import pandas as pd
from src.features.builder import build_training_dataset
from src.model.train import train_model
from src.model.evaluate import run_season_backtest

FOLDS = [
    (list(range(2019, 2022)), 2022),
    (list(range(2019, 2023)), 2023),
    (list(range(2019, 2024)), 2024),
]


def main() -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    model_path = config["model"]["path"]
    db_path = config["db"]["path"]

    all_results = []
    for train_seasons, val_season in FOLDS:
        print(f"\nFold: train={train_seasons} → validate={val_season}")
        train_df = build_training_dataset(db_path=db_path, seasons=train_seasons)
        print(f"  Training on {len(train_df)} games...")
        train_model(train_df, model_path)
        results = run_season_backtest(model_path, [val_season], db_path)
        for r in results:
            r["fold_val_season"] = val_season
        all_results.extend(results)
        fold_df = pd.DataFrame(results)
        print(f"  {val_season}: accuracy={fold_df['accuracy'].mean():.3f} "
              f"brier={fold_df['brier_score'].mean():.4f} "
              f"exp_pts={fold_df['expected_points'].mean():.1f} "
              f"actual_pts={fold_df['actual_points'].mean():.1f}")

    df = pd.DataFrame(all_results)
    print(f"\n=== Walk-forward summary (all folds) ===")
    print(f"  Accuracy:        {df['accuracy'].mean():.3f}  "
          f"(baseline: {df['baseline_accuracy'].mean():.3f})")
    print(f"  Brier score:     {df['brier_score'].mean():.4f}")
    print(f"  Expected pts/wk: {df['expected_points'].mean():.1f}")
    print(f"  Actual pts/wk:   {df['actual_points'].mean():.1f}")

    print(f"\n=== Per-fold summary ===")
    fold_summary = df.groupby("fold_val_season").agg(
        accuracy=("accuracy", "mean"),
        brier_score=("brier_score", "mean"),
        expected_points=("expected_points", "mean"),
        actual_points=("actual_points", "mean"),
    )
    print(fold_summary.to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the backtest end-to-end**

```bash
uv run python scripts/backfill_odds.py   # must have run first to populate game_odds
uv run python scripts/backtest.py
```

Expected: output showing 3 folds, per-fold metrics, and summary table. No crashes.

- [ ] **Step 3: Commit**

```bash
git add scripts/backtest.py
git commit -m "feat: walk-forward backtest with per-fold accuracy, Brier, and confidence-points metrics"
```

---

## Self-Review

**Spec coverage check:**

| Decision | Covered by |
|---|---|
| Market corrector framing | Task 3 — spread is first feature, model learns corrections on top |
| Replace `odds_home_win_prob` with `home_spread` + `game_total` | Task 3 |
| Bet365 from ESPN core API | Task 2 |
| Exclude games with missing odds from training | Task 3 — `build_training_dataset` skips when `get_game_odds` returns None |
| Infer with -1.5/45.5 fallback | Not yet wired — refresh_weekly.py doesn't exist; flag for future API layer task |
| Platt (sigmoid) calibration | Task 4 |
| Walk-forward validation | Task 5 |
| Expected confidence points metric | Already in `evaluate.py:compute_week_metrics` — surfaced by Task 5's backtest output |

**Gap:** Inference fallback (-1.5/45.5 when odds missing) is not wired up because `scripts/refresh_weekly.py` and `src/api/main.py` don't exist yet. When the inference path is built, `fetch_game_odds` should return `None` and the caller should default to `home_spread=-1.5, game_total=45.5` and flag the game as `is_uncertain=True`.

**Placeholder scan:** None found. All steps contain complete code.

**Type consistency:**
- `fetch_game_odds(espn_id: str) -> dict | None` — returns keys `espn_id, home_spread, game_total, home_moneyline, away_moneyline`
- `get_game_odds(db_path, espn_id)` — returns same keys (from DB row)
- `build_features_for_game(game, db_path, home_spread, game_total, weather=None)` — `home_spread` and `game_total` are `float`
- `build_training_dataset(db_path, seasons)` — calls `get_game_odds`, passes `odds["home_spread"]` and `odds["game_total"]` — types match
- `FEATURE_COLS` has 28 entries — test asserts 28 — consistent
