# Seasonal Model Maintenance Plan (2026)

> **Status:** Updated 2026-05-10. Implementation plan with current script inventory and workflow.

**Goal:** Close the data loop by ingesting completed game box scores so the model's 4-week rolling features use real stats, then generate picks for upcoming weeks. Retrain only after the season ends — not mid-season.

**Tech Stack:** Python 3.11, httpx, SQLite, XGBoost, pandas

---

## Weekly Schedule (Workflow)

| Day | Task | Script |
|---|---|---|
| **Tuesday 9:00 AM** | Ingest past week's box scores (closes data loop) | `scripts/ingest_historical.py --weeks 1` |
| **Wednesday 12:00 PM** | Fetch odds/injuries/weather, generate picks | `scripts/refresh_weekly.py` |
| **After season ends** | Retrain model on full history | `scripts/train_production.py` |

**Why no mid-season retraining:** The model is trained on 8 seasons (2,088 games). Adding ~18 completed games from the current season barely moves the needle and risks destabilizing a calibrated model. The weekly refresh is sufficient — it updates features via `team_game_stats` without retraining.

---

## File Map (Current Inventory)

| File | Status | Purpose |
|---|---|---|
| `scripts/ingest_historical.py` | **Done** | Flexible ingestion: full backfill, specific week, or last N weeks. Refreshes box scores for existing games. |
| `scripts/train_production.py` | **Done** | Trains production model on all available seasons. Supports `--seasons` override and auto-discovery. |
| `scripts/backtest.py` | **Done** | Walk-forward evaluation with `--strategy all/recent` and `--compare` modes. |
| `scripts/refresh_weekly.py` | **Done** | Fetches current week's data, builds features, generates predictions and confidence assignments. |
| `scripts/retrain_model.py` | **Not needed** — use `train_production.py` instead. |
| `scripts/ingest_results.py` | **Not needed** — use `ingest_historical.py --weeks N` instead. |
| `ui/admin.py` | **TODO** | Model performance dashboard (see Task 3). Requires metrics logging first. |

---

## Bug Fixes Applied

- **`src/data/espn.py:133`** — ESPN API field is `redZoneAttempts`, not `redZoneAtts`. Caused all red zone stats to be 0, making the feature useless.
- **`scripts/ingest_historical.py`** — Updated to always refresh box scores for existing games (INSERT OR REPLACE), not just new ones.

---

## Task 1: Result Ingestion Script (DONE)

**File:** `scripts/ingest_historical.py`

Three modes:
```bash
# Full backfill all seasons 2018-2025 (~50 min)
uv run python scripts/ingest_historical.py --full

# Specific week
uv run python scripts/ingest_historical.py --week 2026 14

# Last N completed weeks (auto-detects current week)
uv run python scripts/ingest_historical.py --weeks 2
```

Key behaviors:
- Already-ingested games are skipped for game metadata, but box scores/injuries are always refreshed via INSERT OR REPLACE
- Rate-limited to 1 req/sec on game summaries
- `home_win` check skips incomplete/upcoming games during full backfill

---

## Task 2: Retraining Wrapper (DONE)

**File:** `scripts/train_production.py`

```bash
# Auto-discover seasons with completed games + odds
uv run python scripts/train_production.py

# Specific seasons
uv run python scripts/train_production.py --seasons 2021,2022,2023,2024,2025

# From config.yaml model.train_seasons
uv run python scripts/train_production.py --from-config
```

**Not for mid-season use.** See Weekly Schedule above.

---

## Task 3: Admin Dashboard (TODO)

**File:** `ui/admin.py`
**Pre-requisite:** Metrics logging in `train_production.py` or `backtest.py`

The dashboard queries a `model_metrics` table that doesn't get populated yet. Two options:

1. **Add metrics logging to `train_production.py`** — after training, save CV accuracy/Brier score/n_samples to the DB. Simple, one-off metrics per model version.
2. **Add weekly metrics logging to `refresh_weekly.py`** — after generating picks, compare predictions against actual results and log accuracy/Brier/points per week.

Dashboard features:
- Model version, training date, CV accuracy over time
- Weekly prediction accuracy and Brier score trend
- Expected vs actual confidence points tracking
- "Worst misses" table (highest-confidence wrong predictions)

---

## Task 4: Walk-Forward Comparison (DONE)

**File:** `scripts/backtest.py`

```bash
# All seasons (2018+) — default
uv run python scripts/backtest.py

# Recent only (2021+)
uv run python scripts/backtest.py --strategy recent

# Side-by-side comparison
uv run python scripts/backtest.py --compare
```

**Result (as of 2026-05-10):** All-seasons wins on every metric. Recent-only (2021+) has worse Brier score and lower expected points — not enough data (~1,053 vs 2,088 games).

| Metric | All Seasons (2018+) | Recent (2021+) |
|--------|-------------------|----------------|
| Accuracy | 0.687 | 0.685 |
| Brier Score | 0.202 | 0.212 |
| Expected pts/wk | 86.35 | 80.83 |
| Actual pts/wk | 89.81 | 88.35 |

---

## Task 5: Feature Importance Verification (DONE)

Red zone features (`home_red_zone_pct_4wk`, `away_red_zone_pct_4wk`) were 0.0 importance before the fix. After fixing `redZoneAttempts` and retraining:

- **Average importance:** ~0.026 (ranked ~17th of 36 features)
- Mid-tier — comparable to third down %, turnover diff, home/road win pct
- Not a primary driver (spread = 0.12, rest advantage = 0.09)
- Real improvement in walk-forward accuracy, but incremental

---

## Verification Checklist

- [x] `ingest_historical.py` populates `team_game_stats` with correct red zone values
- [x] `build_features_for_game` calculates non-zero `red_zone_pct` from box scores
- [x] `train_production.py` produces valid `model.joblib` with non-zero red zone feature importance
- [x] `backtest.py --compare` shows all-seasons beats recent-only
- [ ] Metrics logging added to `train_production.py` (pre-requisite for admin dashboard)
- [ ] Admin dashboard (`ui/admin.py`) built and tested
