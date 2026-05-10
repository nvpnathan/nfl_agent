# Admin Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add metrics logging to training scripts and build an admin dashboard showing training run history with deep analytics on model vs user picks.

**Architecture:** Add a `model_training_runs` table for training metadata, persist backtest metrics to the existing `model_metrics` ghost table, build admin analytics in a new `ui/admin.py` module integrated into the main app's sidebar. All queries are direct SQLite reads — no new API endpoints needed since admin runs as a Streamlit page.

**Tech Stack:** Python 3.11+, SQLite (via `sqlite3`), Streamlit, plotly (new dependency), joblib, pandas

**Dependencies to add:** `plotly>=6.0` in pyproject.toml

---

### Task 1: Add plotly dependency to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add plotly to dependencies**

In `pyproject.toml`, add `"plotly>=6.0"` after `"pandas>=2.0"` in the dependencies list:

```toml
dependencies = [
    "xgboost>=2.0",
    "scikit-learn>=1.4",
    "pandas>=2.0",
    "plotly>=6.0",
    "numpy>=1.26",
```

- [ ] **Step 2: Install the new dependency**

Run: `uv sync`
Expected: plotly installs without errors.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add plotly for admin dashboard charts"
```

---

### Task 2: Add `model_training_runs` table to schema

**Files:**
- Modify: `src/db/schema.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Add table definition to schema**

In `src/db/schema.py`, add this CREATE TABLE after the `model_metrics` table (after line 175, before the `DROP TABLE IF EXISTS family_picks;`):

```sql
CREATE TABLE IF NOT EXISTS model_training_runs (
    model_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    cv_accuracy_mean REAL NOT NULL,
    cv_accuracy_std REAL NOT NULL,
    n_samples INTEGER NOT NULL,
    seasons_used TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
```

- [ ] **Step 2: Update test to expect the new table**

In `tests/test_db.py`, add `"model_training_runs"` to the expected set in `test_schema_creates_all_tables`:

```python
expected = {
    "games", "team_game_stats", "injury_reports", "depth_charts",
    "predictions", "weekly_assignments", "conversations",
    "rerankings", "model_metrics", "game_odds",
    "weekly_submissions", "weekly_submission_picks",
    "model_training_runs",  # add this line
}
```

- [ ] **Step 3: Run schema tests**

Run: `uv run pytest tests/test_db.py::test_schema_creates_all_tables -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/db/schema.py tests/test_db.py
git commit -m "db: add model_training_runs table for training run metadata"
```

---

### Task 3: Add query functions for training runs and admin analytics

**Files:**
- Modify: `src/db/queries.py`

- [ ] **Step 1: Add all query functions**

Append to `src/db/queries.py`:

```python
def insert_model_training_run(db_path: str, run: dict) -> int:
    """Insert a training run record. Returns the new model_run_id."""
    with _conn(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO model_training_runs
                (model_version, cv_accuracy_mean, cv_accuracy_std, n_samples, seasons_used)
            VALUES (?, ?, ?, ?, ?)
        """, (run["model_version"], run["cv_accuracy_mean"], run["cv_accuracy_std"],
              run["n_samples"], run["seasons_used"]))
        return cursor.lastrowid


def get_model_training_runs(db_path: str) -> list[dict]:
    """Get all training runs, ordered by created_at descending."""
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT * FROM model_training_runs ORDER BY created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_seasons_with_submissions(db_path: str) -> list[int]:
    """Get all seasons that have at least one weekly submission."""
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT DISTINCT season FROM weekly_submissions ORDER BY season DESC
        """).fetchall()
    return [int(r[0]) for r in rows]


def get_weekly_analytics(db_path: str, season: int, week: int) -> dict | None:
    """Get analytics data for a specific season/week.

    Returns dict with keys: 'picks', 'overrides'.
    Each pick has: game_id, home_team, away_team, pick (team name), source ('model' or 'override'),
    pick_correct (0/1), model_correct (0/1), model_confidence, submitted_points.
    Returns None if no submission exists.
    """
    with _conn(db_path) as conn:

        sub = conn.execute("""
            SELECT submission_id FROM weekly_submissions WHERE season=? AND week=?
        """, (season, week)).fetchone()
        if sub is None:
            return None

        submission_id = sub["submission_id"]

        picks_rows = conn.execute("""
            SELECT
                sp.game_id,
                g.home_team,
                g.away_team,
                CASE WHEN sp.submitted_pick = g.home_team THEN 'home' ELSE 'away' END AS pick_side,
                sp.submitted_pick AS picked_team,
                p.predicted_winner AS model_picked_team,
                sp.is_overridden,
                g.home_win,
                p.home_win_prob AS model_confidence,
                sp.submitted_points,
                CASE WHEN sp.submitted_pick = g.home_team AND g.home_win = 1 THEN 1
                     WHEN sp.submitted_pick = g.away_team AND g.home_win = 0 THEN 1
                     ELSE 0 END AS pick_correct,
                CASE WHEN p.predicted_winner = g.home_team AND g.home_win = 1 THEN 1
                     WHEN p.predicted_winner = g.away_team AND g.home_win = 0 THEN 1
                     ELSE 0 END AS model_correct
            FROM weekly_submission_picks sp
            JOIN games g ON sp.game_id = g.espn_id
            LEFT JOIN predictions p ON sp.game_id = p.game_id
                AND p.season=? AND p.week=?
            WHERE sp.submission_id=?
            ORDER BY sp.submitted_points DESC, g.game_date ASC
        """, (season, week, submission_id)).fetchall()

    picks = [dict(r) for r in picks_rows]
    if not picks:
        return None

    # Add source tag
    for p in picks:
        p["source"] = "override" if p["is_overridden"] else "model"

    overrides = [p for p in picks if p["is_overridden"]]

    return {"picks": picks, "overrides": overrides}


def get_season_override_summary(db_path: str, season: int) -> list[dict]:
    """Get override summary per week for a season.

    Returns list of dicts with: 'week', 'overrides_made', 'saved', 'hurt'.
    """
    with _conn(db_path) as conn:
        weeks = [int(r[0]) for r in conn.execute("""
            SELECT DISTINCT week FROM weekly_submissions WHERE season=? ORDER BY week
        """, (season,)).fetchall()]

    results = []
    for week in weeks:
        with _conn(db_path) as conn:
            sub = conn.execute("""
                SELECT submission_id FROM weekly_submissions WHERE season=? AND week=?
            """, (season, week)).fetchone()
        if sub is None:
            continue

        rows = conn.execute("""
            SELECT
                CASE WHEN sp.submitted_pick = g.home_team AND g.home_win = 1 THEN 1
                     WHEN sp.submitted_pick = g.away_team AND g.home_win = 0 THEN 1
                     ELSE 0 END AS user_correct,
                CASE WHEN p.predicted_winner = g.home_team AND g.home_win = 1 THEN 1
                     WHEN p.predicted_winner = g.away_team AND g.home_win = 0 THEN 1
                     ELSE 0 END AS model_correct
            FROM weekly_submission_picks sp
            JOIN games g ON sp.game_id = g.espn_id
            LEFT JOIN predictions p ON sp.game_id = p.game_id
                AND p.season=? AND p.week=?
            WHERE sp.submission_id=? AND sp.is_overridden=1
        """, (season, week, sub["submission_id"])).fetchall()

        total = len(rows)
        if total == 0:
            results.append({"week": week, "overrides_made": 0, "saved": 0, "hurt": 0})
            continue

        saved = sum(1 for r in rows if int(r["user_correct"]) == 1 and int(r["model_correct"]) == 0)
        hurt = sum(1 for r in rows if int(r["user_correct"]) == 0 and int(r["model_correct"]) == 1)

        results.append({
            "week": week,
            "overrides_made": total,
            "saved": saved,
            "hurt": hurt,
        })

    return results


def get_weekly_overall_stats(db_path: str, season: int) -> dict | None:
    """Get overall stats for a season across all weeks.

    Returns dict with keys: 'weeks_covered', 'actual_pct', 'model_pct',
    'overrides_total', 'saved', 'hurt'. Returns None if no data.
    """
    with _conn(db_path) as conn:
        weeks = [int(r[0]) for r in conn.execute("""
            SELECT DISTINCT week FROM weekly_submissions WHERE season=? ORDER BY week
        """, (season,)).fetchall()]

    if not weeks:
        return None

    rows = conn.execute("""
        SELECT
            sp.submitted_pick,
            g.home_team,
            g.away_team,
            g.home_win,
            p.predicted_winner AS model_picked,
            sp.is_overridden
        FROM weekly_submission_picks sp
        JOIN games g ON sp.game_id = g.espn_id
        LEFT JOIN predictions p ON sp.game_id = p.game_id
            AND p.season=? AND p.week=?
        WHERE sp.submission_id IN (SELECT submission_id FROM weekly_submissions WHERE season=?)
    """, (season, season, season)).fetchall()

    total = len(rows)
    if total == 0:
        return None

    actual_correct = sum(
        1 for r in rows
        if (r["submitted_pick"] == r["home_team"] and r["home_win"] == 1)
        or (r["submitted_pick"] == r["away_team"] and r["home_win"] == 0)
    )
    model_correct = sum(
        1 for r in rows
        if (r["model_picked"] == r["home_team"] and r["home_win"] == 1)
        or (r["model_picked"] == r["away_team"] and r["home_win"] == 0)
    )

    override_rows = [r for r in rows if int(r["is_overridden"]) == 1]
    om = len(override_rows)
    saved = sum(
        1 for r in override_rows
        if ((r["submitted_pick"] == r["home_team"] and r["home_win"] == 1)
            or (r["submitted_pick"] == r["away_team"] and r["home_win"] == 0))
        and not ((r["model_picked"] == r["home_team"] and r["home_win"] == 1)
                 or (r["model_picked"] == r["away_team"] and r["home_win"] == 0))
    )
    hurt = sum(
        1 for r in override_rows
        if not ((r["submitted_pick"] == r["home_team"] and r["home_win"] == 1)
                or (r["submitted_pick"] == r["away_team"] and r["home_win"] == 0))
        and ((r["model_picked"] == r["home_team"] and r["home_win"] == 1)
             or (r["model_picked"] == r["away_team"] and r["home_win"] == 0))
    )

    return {
        "weeks_covered": len(weeks),
        "actual_pct": round(actual_correct / total, 3),
        "model_pct": round(model_correct / total, 3),
        "overrides_total": om,
        "saved": saved,
        "hurt": hurt,
    }
```

- [ ] **Step 2: Run existing db tests to ensure no regressions**

Run: `uv run pytest tests/test_db.py -v`
Expected: All existing tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/db/queries.py tests/test_db.py
git commit -m "db: add query functions for training runs and admin analytics"
```

---

### Task 4: Persist backtest metrics to `model_metrics` table

**Files:**
- Modify: `src/model/evaluate.py`
- Modify: `scripts/backtest.py`

- [ ] **Step 1: Add persist_week_metrics function to evaluate.py**

Append to `src/model/evaluate.py`:

```python
def persist_week_metrics(db_path: str, metrics_list: list[dict]) -> int:
    """Insert each week's metrics into model_metrics table. Returns count inserted."""
    from src.db.queries import _conn as db_conn
    from src.model.train import MODEL_VERSION

    with db_conn(db_path) as conn:
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

- [ ] **Step 2: Add --persist flag to backtest.py**

In `scripts/backtest.py`, add after the existing argument definitions (after line 160):

```python
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist backtest metrics to the model_metrics table.",
    )
```

In `main()`, after line 95 (after fold summary print), before the `finally` block:

```python
        if args.persist:
            from src.model.evaluate import persist_week_metrics
            count = persist_week_metrics(db_path, all_results)
            print(f"Persisted {count} week metrics to model_metrics table.")
```

- [ ] **Step 3: Run to verify**

Run: `uv run python scripts/backtest.py --help`
Expected: Shows `--persist` flag in output.

- [ ] **Step 4: Commit**

```bash
git add src/model/evaluate.py scripts/backtest.py
git commit -m "eval: persist backtest metrics to model_metrics table with --persist flag"
```

---

### Task 5: Train production logging — insert training run after each training

**Files:**
- Modify: `scripts/train_production.py`

- [ ] **Step 1: Add training run persistence**

Add `import json` at the top of `scripts/train_production.py` (after `import argparse`).

After line 80 (`metrics = train_model(df, model_path)`), replace lines 81-82:

```python
    metrics = train_model(df, model_path)
    run_id = insert_model_training_run(db_path, {
        "model_version": metrics["model_version"],
        "cv_accuracy_mean": metrics["cv_accuracy_mean"],
        "cv_accuracy_std": metrics["cv_accuracy_std"],
        "n_samples": metrics["n_samples"],
        "seasons_used": json.dumps(seasons),
    })
    print(f"Training run recorded: id={run_id}")
    print("Training complete:")
    print(metrics)
```

Add `from src.db.queries import insert_model_training_run` at the top of the file (after existing imports).

- [ ] **Step 2: Run to verify no import errors**

Run: `uv run python scripts/train_production.py --help`
Expected: Help text prints, no import errors.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_production.py
git commit -m "train: log training runs to model_training_runs table"
```

---

### Task 6: Build admin dashboard module (`ui/admin.py`)

**Files:**
- Create: `ui/admin.py`

This is a standalone module with functions that can be called from `ui/app.py` or run independently.

- [ ] **Step 1: Create ui/admin.py**

Create `ui/admin.py` with this exact content:

```python
"""NFL Confidence Pool — Admin dashboard.

Shows training run history and deep analytics on model vs user picks.
Import from ui/app.py sidebar, or run standalone: uv run streamlit run ui/admin.py
"""
import json
import sys
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.db.queries import (
    get_model_training_runs,
    get_seasons_with_submissions,
    get_weekly_analytics,
    get_season_override_summary,
    get_weekly_overall_stats,
)
from src.db.queries import _conn as db_conn


def render_admin(season: int | None = None, week: int | None = None):
    """Render the admin dashboard. Called from ui/app.py sidebar."""

    with st.sidebar:
        if st.button("← Back to Picks", use_container_width=True):
            st.session_state["admin_mode"] = False
            st.rerun()

    # Get available seasons
    try:
        seasons = get_seasons_with_submissions(_db_path())
        if not seasons:
            conn = db_conn(_db_path())
            rows = conn.execute(
                "SELECT DISTINCT season FROM games ORDER BY season DESC"
            ).fetchall()
            seasons = [int(r[0]) for r in rows]
            conn.close()
    except Exception:
        seasons = []

    if not seasons:
        st.info("No data available yet. Train a model or run a backtest first.")
        return

    if season is None:
        season = seasons[0]
    elif season not in seasons:
        season = seasons[0]

    admin_season = st.selectbox("Season", seasons, key="admin_season")
    tab1, tab2 = st.tabs(["Training Runs", "Historical Results"])

    with tab1:
        _render_training_runs_tab()

    with tab2:
        _render_historical_results_tab(admin_season)


def _db_path() -> str:
    """Get database path from config."""
    import yaml
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        config = yaml.safe_load(f)
    return config["db"]["path"]


def _render_training_runs_tab():
    """Render the training runs tab with chart and table."""
    runs = get_model_training_runs(_db_path())

    if not runs:
        st.info("No training runs recorded yet. Run `train_production.py` to train a model.")
        return

    df = pd.DataFrame(runs)
    df["seasons_list"] = df["seasons_used"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    # Chart: CV accuracy trend over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["created_at"],
        y=df["cv_accuracy_mean"],
        mode="lines+markers",
        name="CV Accuracy",
        marker=dict(size=8, color="#4ADE80"),
        line=dict(width=2, color="#4ADE80"),
    ))
    fig.update_layout(
        title="Model CV Accuracy Over Time",
        xaxis_title="Training Date",
        yaxis_title="CV Accuracy (Mean)",
        height=350,
        template="plotly_dark",
    )
    st.plotly_chart(fig)

    # Table: all runs
    st.subheader("Training Run History")
    table_data = []
    for r in runs:
        seasons_str = ", ".join(str(s) for s in json.loads(r["seasons_used"]))
        table_data.append({
            "Date": r["created_at"][:10],
            "Version": r["model_version"],
            "CV Accuracy": f"{r['cv_accuracy_mean']:.3f} ± {r['cv_accuracy_std']:.3f}",
            "Samples": r["n_samples"],
            "Seasons": seasons_str,
        })

    st.dataframe(table_data, use_container_width=True, hide_index=True)


def _render_historical_results_tab(season: int):
    """Render the historical results tab with analytics."""

    overall = get_weekly_overall_stats(_db_path(), season)

    if not overall:
        st.info(f"No submission data found for {season}.")
        return

    # Summary cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Weeks Covered", overall["weeks_covered"])
    with c2:
        st.metric("Actual Result %", f"{overall['actual_pct']:.1%}")
    with c3:
        st.metric("Model Prediction %", f"{overall['model_pct']:.1%}")

    if overall["overrides_total"] > 0:
        c4, c5 = st.columns(2)
        with c4:
            st.metric("Overrides Saved You", overall["saved"])
        with c5:
            st.metric("Overrides Hurt You", overall["hurt"])

    # Week selector from override summary
    summary = get_season_override_summary(_db_path(), season)
    weeks_with_data = sorted(set(s["week"] for s in summary))

    if not weeks_with_data:
        st.info(f"No data for {season}.")
        return

    selected_week = st.selectbox("Week", weeks_with_data, key="admin_week")

    analytics = get_weekly_analytics(_db_path(), season, selected_week)

    if not analytics or not analytics["picks"]:
        st.info(f"No pick data for {season} Week {selected_week}.")
        return

    picks = analytics["picks"]
    overrides = [p for p in picks if p["is_overridden"]]

    # Section A: All Picks
    st.subheader(f"All Picks — Week {selected_week}")

    picks_rows = []
    for p in picks:
        correct_icon = "✅" if p["pick_correct"] else "❌"
        conf_str = f"{p['model_confidence']:.0%}" if p["model_confidence"] is not None else "—"
        picks_rows.append({
            f"{p['away_team']} @ {p['home_team']}": "",
            "Pick": p["picked_team"],
            "Source": f"({p['source']})",
            "Result": correct_icon,
            "Model Conf.": conf_str,
            "Conf Pts": p["submitted_points"],
        })

    st.dataframe(picks_rows, use_container_width=True, hide_index=True)

    # Section C: Override Detail
    if overrides:
        st.subheader(f"Override Detail — Week {selected_week}")

        saved = sum(
            1 for o in overrides
            if int(o["pick_correct"]) == 1 and int(o["model_correct"]) == 0
        )
        hurt = sum(
            1 for o in overrides
            if int(o["pick_correct"]) == 0 and int(o["model_correct"]) == 1
        )

        st.caption(f"Overrides: {len(overrides)} made · ✅ Saved you: {saved} · ❌ Hurt you: {hurt}")

        override_rows = []
        for o in overrides:
            if int(o["pick_correct"]) == 1 and int(o["model_correct"]) == 0:
                verdict = "✅ Override saved you"
            else:
                verdict = "❌ Override hurt you"

            override_rows.append({
                f"{o['away_team']} @ {o['home_team']}": "",
                "Model Correct?": "✅" if int(o["model_correct"]) else "❌",
                "User Correct?": "✅" if int(o["pick_correct"]) else "❌",
                "Verdict": verdict,
            })

        st.dataframe(override_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No overrides this week — user agreed with model on all picks.")

    # Section D: Season Override Breakdown
    st.subheader("Season Override Breakdown")

    if not any(s["overrides_made"] > 0 for s in summary):
        st.caption("No overrides made this season.")
    else:
        fig = go.Figure()
        weeks_list = [s["week"] for s in summary if s["overrides_made"] > 0]
        saved_list = [s["saved"] for s in summary if s["overrides_made"] > 0]
        hurt_list = [s["hurt"] for s in summary if s["overrides_made"] > 0]

        fig.add_trace(go.Bar(
            x=weeks_list, y=saved_list, name="Saved You",
            marker_color="#4ADE80",
        ))
        fig.add_trace(go.Bar(
            x=weeks_list, y=hurt_list, name="Hurt You",
            marker_color="#F87171",
        ))
        fig.update_layout(
            title="Overrides: Saved vs Hurt by Week",
            xaxis_title="Week",
            yaxis_title="Count",
            barmode="stack",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

        table_data = [
            {
                "Week": s["week"],
                "Overrides Made": s["overrides_made"],
                "Saved You": s["saved"],
                "Hurt You": s["hurt"],
            }
            for s in summary if s["overrides_made"] > 0
        ]

        st.dataframe(table_data, use_container_width=True, hide_index=True)
```

- [ ] **Step 2: Verify the module imports correctly**

Run: `uv run python -c "from ui.admin import render_admin; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add ui/admin.py
git commit -m "ui: add admin dashboard with training runs and historical results analytics"
```

---

### Task 7: Integrate admin into `ui/app.py` sidebar

**Files:**
- Modify: `ui/app.py`

- [ ] **Step 1: Add admin tab to sidebar**

In `ui/app.py`, after line 57 (after the API connection status text, still inside the `with st.sidebar:` block), add:

```python
    st.divider()
    if st.button("Admin", type="secondary", use_container_width=True, key="sb_admin"):
        st.session_state["admin_mode"] = True
        st.rerun()
```

Then, after the sidebar block ends (after line 57), before the data loading section (before line 60, which starts with `# ── data & header ──`), add:

```python
# ── admin mode check ─────────────────────────────────────────────────────────────

if st.session_state.get("admin_mode", False):
    from ui.admin import render_admin
    render_admin()
    st.stop()
```

- [ ] **Step 2: Verify the integration works**

Run: `uv run streamlit run ui/app.py --server.headless true`
Expected: App starts without errors.

- [ ] **Step 3: Commit**

```bash
git add ui/app.py
git commit -m "ui: integrate admin dashboard into app sidebar"
```

---

### Task 8: Add tests for new query functions and training run logging

**Files:**
- Create: `tests/test_admin.py`

- [ ] **Step 1: Add comprehensive tests**

Create `tests/test_admin.py`:

```python
import os
import json
import pytest
from src.db.schema import create_schema


@pytest.fixture
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    create_schema(db_path)
    return db_path


def test_insert_model_training_run(tmp_db):
    from src.db.queries import insert_model_training_run, get_model_training_runs

    run_id = insert_model_training_run(tmp_db, {
        "model_version": "xgb_v1",
        "cv_accuracy_mean": 0.675,
        "cv_accuracy_std": 0.023,
        "n_samples": 4567,
        "seasons_used": json.dumps([2018, 2019, 2020, 2021, 2022, 2023]),
    })

    assert run_id == 1

    runs = get_model_training_runs(tmp_db)
    assert len(runs) == 1
    assert runs[0]["model_version"] == "xgb_v1"
    assert runs[0]["cv_accuracy_mean"] == 0.675


def test_get_model_training_runs_empty(tmp_db):
    from src.db.queries import get_model_training_runs

    runs = get_model_training_runs(tmp_db)
    assert runs == []


def test_get_seasons_with_submissions(tmp_db):
    from src.db.queries import (
        get_seasons_with_submissions, insert_espn_game, upsert_weekly_assignment,
        create_weekly_submission,
    )

    insert_espn_game(tmp_db, {
        "espn_id": "g1", "season": 2024, "week": 1, "game_type": "regular",
        "home_team": "KC", "away_team": "BUF", "home_espn_id": "12",
        "away_espn_id": "33", "game_date": "2024-09-05T20:00Z",
        "venue": "X", "is_indoor": 0, "is_neutral": 0,
        "attendance": None, "home_score": None, "away_score": None, "home_win": None,
    })
    upsert_weekly_assignment(tmp_db, {
        "season": 2024, "week": 1, "game_id": "g1",
        "predicted_winner": "KC", "confidence_points": 2,
        "win_probability": 0.75, "is_uncertain": 0,
        "is_overridden": 0, "override_reason": None,
    })

    create_weekly_submission(tmp_db, 2024, 1, source="test")

    seasons = get_seasons_with_submissions(tmp_db)
    assert 2024 in seasons


def test_get_seasons_with_submissions_empty(tmp_db):
    from src.db.queries import get_seasons_with_submissions

    seasons = get_seasons_with_submissions(tmp_db)
    assert seasons == []


def test_get_weekly_overall_stats(tmp_db):
    from src.db.queries import (
        get_weekly_overall_stats, insert_espn_game, upsert_weekly_assignment,
        create_weekly_submission, swap_confidence_points, insert_prediction,
    )

    # Game 1: home (KC) wins. Model picks KC (correct). User picks KC (correct, no override).
    insert_espn_game(tmp_db, {
        "espn_id": "g1", "season": 2024, "week": 1, "game_type": "regular",
        "home_team": "KC", "away_team": "BUF", "home_espn_id": "12",
        "away_espn_id": "33", "game_date": "2024-09-05T20:00Z",
        "venue": "X", "is_indoor": 0, "is_neutral": 0,
        "attendance": None, "home_score": 27, "away_score": 20, "home_win": 1,
    })

    # Game 2: away (PHI) wins. Model picks DAL (wrong). User overrides to PHI (correct, saved!).
    insert_espn_game(tmp_db, {
        "espn_id": "g2", "season": 2024, "week": 1, "game_type": "regular",
        "home_team": "DAL", "away_team": "PHI", "home_espn_id": "21",
        "away_espn_id": "28", "game_date": "2024-09-12T20:00Z",
        "venue": "X", "is_indoor": 0, "is_neutral": 0,
        "attendance": None, "home_score": 19, "away_score": 27, "home_win": 0,
    })

    # Model assigns points: KC=2 (higher prob), DAL=1 (lower prob)
    upsert_weekly_assignment(tmp_db, {
        "season": 2024, "week": 1, "game_id": "g1",
        "predicted_winner": "KC", "confidence_points": 2,
        "win_probability": 0.75, "is_uncertain": 0,
        "is_overridden": 0, "override_reason": None,
    })
    upsert_weekly_assignment(tmp_db, {
        "season": 2024, "week": 1, "game_id": "g2",
        "predicted_winner": "DAL", "confidence_points": 1,
        "win_probability": 0.55, "is_uncertain": 0,
        "is_overridden": 0, "override_reason": None,
    })

    # User overrides g2 from DAL to PHI (model was wrong, user right = saved)
    swap_confidence_points(tmp_db, 2024, 1, "g2", 2, "user override")

    # Insert predictions for model comparison
    insert_prediction(tmp_db, {
        "game_id": "g1", "season": 2024, "week": 1,
        "home_win_prob": 0.75, "odds_implied_prob": None,
        "model_version": "xgb_v1", "predicted_winner": "KC",
    })
    insert_prediction(tmp_db, {
        "game_id": "g2", "season": 2024, "week": 1,
        "home_win_prob": 0.55, "odds_implied_prob": None,
        "model_version": "xgb_v1", "predicted_winner": "DAL",
    })

    create_weekly_submission(tmp_db, 2024, 1, source="test")

    stats = get_weekly_overall_stats(tmp_db, 2024)
    assert stats is not None
    assert stats["weeks_covered"] == 1
    # User: KC correct (2/2 = 100%), Model: KC correct, DAL wrong (1/2 = 50%)
    assert stats["actual_pct"] == 1.0   # user got both right (KC model pick, PHI override)
    assert stats["model_pct"] == 0.5   # model got KC right, DAL wrong
    assert stats["overrides_total"] == 1
    assert stats["saved"] == 1  # model wrong on g2, user right


def test_get_season_override_summary_saved(tmp_db):
    """Test override that saved the user (model wrong, user right)."""
    from src.db.queries import (
        get_season_override_summary, insert_espn_game, upsert_weekly_assignment,
        create_weekly_submission, swap_confidence_points, insert_prediction,
    )

    # Model picks BUF (away), KC actually wins (home). Model wrong.
    insert_espn_game(tmp_db, {
        "espn_id": "g1", "season": 2024, "week": 1, "game_type": "regular",
        "home_team": "KC", "away_team": "BUF", "home_espn_id": "12",
        "away_espn_id": "33", "game_date": "2024-09-05T20:00Z",
        "venue": "X", "is_indoor": 0, "is_neutral": 0,
        "attendance": None, "home_score": 27, "away_score": 20, "home_win": 1,
    })

    # Model assigns BUF with high points (wrong pick)
    upsert_weekly_assignment(tmp_db, {
        "season": 2024, "week": 1, "game_id": "g1",
        "predicted_winner": "BUF", "confidence_points": 2,
        "win_probability": 0.75, "is_uncertain": 0,
        "is_overridden": 0, "override_reason": None,
    })

    # User overrides BUF -> KC (model was wrong, user right = saved)
    swap_confidence_points(tmp_db, 2024, 1, "g1", 1, "user override")

    insert_prediction(tmp_db, {
        "game_id": "g1", "season": 2024, "week": 1,
        "home_win_prob": 0.75, "odds_implied_prob": None,
        "model_version": "xgb_v1", "predicted_winner": "BUF",
    })

    create_weekly_submission(tmp_db, 2024, 1, source="test")

    result = get_season_override_summary(tmp_db, 2024)
    assert len(result) == 1
    assert result[0]["week"] == 1
    assert result[0]["overrides_made"] == 1
    assert result[0]["saved"] == 1


def test_get_season_override_summary_hurt(tmp_db):
    """Test override that hurt the user (model right, user wrong)."""
    from src.db.queries import (
        get_season_override_summary, insert_espn_game, upsert_weekly_assignment,
        create_weekly_submission, swap_confidence_points, insert_prediction,
    )

    # Model picks KC (home), KC actually wins. Model right.
    insert_espn_game(tmp_db, {
        "espn_id": "g1", "season": 2024, "week": 1, "game_type": "regular",
        "home_team": "KC", "away_team": "BUF", "home_espn_id": "12",
        "away_espn_id": "33", "game_date": "2024-09-05T20:00Z",
        "venue": "X", "is_indoor": 0, "is_neutral": 0,
        "attendance": None, "home_score": 27, "away_score": 20, "home_win": 1,
    })

    # Model assigns KC with high points (correct pick)
    upsert_weekly_assignment(tmp_db, {
        "season": 2024, "week": 1, "game_id": "g1",
        "predicted_winner": "KC", "confidence_points": 2,
        "win_probability": 0.75, "is_uncertain": 0,
        "is_overridden": 0, "override_reason": None,
    })

    # User overrides KC -> BUF (model was right, user wrong = hurt)
    swap_confidence_points(tmp_db, 2024, 1, "g1", 1, "user override")

    insert_prediction(tmp_db, {
        "game_id": "g1", "season": 2024, "week": 1,
        "home_win_prob": 0.75, "odds_implied_prob": None,
        "model_version": "xgb_v1", "predicted_winner": "KC",
    })

    create_weekly_submission(tmp_db, 2024, 1, source="test")

    result = get_season_override_summary(tmp_db, 2024)
    assert len(result) == 1
    assert result[0]["week"] == 1
    assert result[0]["overrides_made"] == 1
    assert result[0]["hurt"] == 1


def test_persist_week_metrics():
    from src.db.schema import create_schema
    from src.model.evaluate import persist_week_metrics
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        create_schema(db_path)
        metrics_list = [
            {"season": 2024, "week": 1, "accuracy": 0.75, "brier_score": 0.21,
             "expected_points": 85.5, "actual_points": 90, "baseline_accuracy": 0.62},
            {"season": 2024, "week": 2, "accuracy": 0.69, "brier_score": 0.18,
             "expected_points": 78.0, "actual_points": 82, "baseline_accuracy": 0.62},
        ]

        count = persist_week_metrics(db_path, metrics_list)
        assert count == 2

        import sqlite3
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT * FROM model_metrics").fetchall()
        assert len(rows) == 2
        conn.close()
    finally:
        os.unlink(db_path)


def test_persist_week_metrics_empty(tmp_db):
    from src.model.evaluate import persist_week_metrics

    count = persist_week_metrics(tmp_db, [])
    assert count == 0


def test_get_weekly_analytics_with_overrides(tmp_db):
    from src.db.queries import (
        get_weekly_analytics, insert_espn_game, upsert_weekly_assignment,
        create_weekly_submission, swap_confidence_points, insert_prediction,
    )

    # Game: away (BUF) wins. Model picks KC (wrong). User overrides to BUF (right = saved).
    insert_espn_game(tmp_db, {
        "espn_id": "g1", "season": 2024, "week": 1, "game_type": "regular",
        "home_team": "KC", "away_team": "BUF", "home_espn_id": "12",
        "away_espn_id": "33", "game_date": "2024-09-05T20:00Z",
        "venue": "X", "is_indoor": 0, "is_neutral": 0,
        "attendance": None, "home_score": 20, "away_score": 27, "home_win": 0,
    })

    upsert_weekly_assignment(tmp_db, {
        "season": 2024, "week": 1, "game_id": "g1",
        "predicted_winner": "KC", "confidence_points": 2,
        "win_probability": 0.75, "is_uncertain": 0,
        "is_overridden": 0, "override_reason": None,
    })

    swap_confidence_points(tmp_db, 2024, 1, "g1", 1, "user override")

    insert_prediction(tmp_db, {
        "game_id": "g1", "season": 2024, "week": 1,
        "home_win_prob": 0.75, "odds_implied_prob": None,
        "model_version": "xgb_v1", "predicted_winner": "KC",
    })

    create_weekly_submission(tmp_db, 2024, 1, source="test")

    result = get_weekly_analytics(tmp_db, 2024, 1)
    assert result is not None
    assert len(result["picks"]) == 1
    pick = result["picks"][0]
    assert pick["picked_team"] == "BUF"  # user overrode to away team
    assert pick["source"] == "override"
    assert int(pick["pick_correct"]) == 1   # user right (BUF away won)
    assert int(pick["model_correct"]) == 0  # model wrong (KC home lost)
    assert len(result["overrides"]) == 1


def test_get_weekly_analytics_no_submission(tmp_db):
    from src.db.queries import get_weekly_analytics

    result = get_weekly_analytics(tmp_db, 2024, 1)
    assert result is None
```

- [ ] **Step 2: Run all tests**

Run: `uv run pytest tests/test_admin.py -v`
Expected: All new tests pass.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass (new + existing).

- [ ] **Step 4: Commit**

```bash
git add tests/test_admin.py
git commit -m "tests: add tests for training run logging, admin analytics queries"
```

---

### Task 9: End-to-end verification

**Files:** none (integration test via CLI)

- [ ] **Step 1: Verify training run logging works end-to-end**

Run: `uv run python scripts/train_production.py --from-config`
Expected: Training completes, prints "Training run recorded: id=N"

- [ ] **Step 2: Verify backtest persistence works**

Run: `uv run python scripts/backtest.py --persist`
Expected: Backtest completes, prints "Persisted N week metrics to model_metrics table."

- [ ] **Step 3: Verify admin dashboard loads**

Run: `uv run streamlit run ui/app.py --server.headless true`
Expected: App loads, sidebar shows "Admin" button.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: admin dashboard with training run logging and pick analytics"
```
