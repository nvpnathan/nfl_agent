# CLAUDE.md

## Project: NFL Confidence Pool Agent

A local AI agent that predicts NFL game winners and assigns confidence points (1–N) optimally for a family season-long confidence pool. Full implementation plan at `docs/superpowers/plans/2026-05-07-nfl-confidence-pool.md`.

### Tech Stack
- **Backend:** FastAPI + Uvicorn (`src/api/main.py`)
- **UI:** Streamlit (`ui/app.py`)
- **Database:** SQLite at `data/nfl_pool.db` (`src/db/`)
- **ML Model:** XGBoost + sklearn calibration (`src/model/`)
- **Agent LLM:** Ollama (local 70B, default) or Claude Sonnet 4.6 — set via `config.yaml`
- **Data:** nfl_data_py (historical), The Odds API, Sleeper API, Open-Meteo

### Key Commands
```bash
# Install dependencies
uv sync

# One-time: load 2018–2025 historical data into SQLite
uv run python scripts/ingest_historical.py

# Train model + run backtest (2024–2025 validation)
uv run python scripts/backtest.py

# Start API server (port 8000)
uv run uvicorn src.api.main:app --reload --port 8000

# Start Streamlit dashboard (port 8501)
uv run streamlit run ui/app.py

# Manual weekly refresh (also runs via cron)
uv run python scripts/refresh_weekly.py

# Run all tests
uv run pytest tests/ -v
```

### Architecture Notes
- `FEATURE_COLS` in `src/features/builder.py` is the single source of truth for model features — all training, prediction, and evaluation code imports from there
- `config.yaml` controls LLM provider, API keys, model paths, and uncertainty threshold
- The confidence optimizer (`src/optimizer/confidence.py`) sorts games by win probability descending; highest probability gets the most points (rearrangement inequality). Games within 3% of each other are flagged as uncertain
- Re-rankings are recorded in the `rerankings` table and surfaced as diffs in the UI
- The agent runs a tool-use loop: it calls tools, gets results, reasons, calls more tools until it has a final answer
- Playoff detection is schedule-driven — `scripts/refresh_weekly.py` looks up the current week's game type from nfl_data_py to determine the correct point range

### Config
```yaml
llm:
  provider: "ollama"      # Switch to "claude" to use Sonnet 4.6
  ollama_model: "llama3.3:70b"
  claude_model: "claude-sonnet-4-6"
model:
  uncertainty_threshold: 0.03   # Flag picks within 3% of each other
```

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Task Delegation

Spawn subagents to isolate context, parallelize independent work, or offload bulk mechanical tasks. Don't spawn when the parent needs the reasoning, when synthesis requires holding things together, or when spawn overhead dominates.

Pick the cheapest model that can do the subtask well:
- Haiku: bulk mechanical work, no judgment
- Sonnet: scoped research, code exploration, in-scope synthesis
- Opus: subtasks needing real planning or tradeoffs

If a subagent realizes it needs a higher tier than itself, return to the parent.

Parent owns final output and cross-spawn synthesis. User instructions override.