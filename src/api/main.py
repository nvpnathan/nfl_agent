import os
import subprocess
import sys
import yaml
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from src.db.schema import create_schema
from src.db.queries import (
    get_games_for_week, get_weekly_assignments, get_injuries_for_week,
    upsert_weekly_assignment, swap_confidence_points,
    create_weekly_submission, get_weekly_submission, revert_assignment_to_model,
)


def _db_path() -> str:
    return os.environ.get("NFL_DB_PATH", "data/nfl_pool.db")


def _config() -> dict:
    with open(os.environ.get("NFL_CONFIG_PATH", "config.yaml")) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_schema(_db_path())
    yield


app = FastAPI(title="NFL Confidence Pool Agent", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/week/{season}/{week}")
def get_week(season: int, week: int):
    db = _db_path()
    assignments = get_weekly_assignments(db, season, week)
    games = get_games_for_week(db, season, week)
    return {"season": season, "week": week, "games": games, "assignments": assignments}


@app.get("/injuries/{season}/{week}/{team}")
def get_injuries(season: int, week: int, team: str):
    return get_injuries_for_week(_db_path(), team, season, week)


class OverrideRequest(BaseModel):
    game_id: str
    confidence_points: int
    reason: str | None = None


class LockRequest(BaseModel):
    source: str = "api"


class RevertRequest(BaseModel):
    game_id: str


@app.post("/week/{season}/{week}/override")
def override_pick(season: int, week: int, req: OverrideRequest):
    """Swap confidence points: sets game_id to new points, swapping with whoever held them."""
    db = _db_path()
    assignments = get_weekly_assignments(db, season, week)
    if not any(a["game_id"] == req.game_id for a in assignments):
        raise HTTPException(status_code=404, detail="Assignment not found")

    displaced_id, old_pts = swap_confidence_points(
        db, season, week, req.game_id, req.confidence_points, req.reason
    )
    return {
        "ok": True,
        "game_id": req.game_id,
        "confidence_points": req.confidence_points,
        "displaced_game_id": displaced_id,
        "displaced_old_points": old_pts,
    }


@app.post("/week/{season}/{week}/revert")
def revert_pick(season: int, week: int, req: RevertRequest):
    try:
        return revert_assignment_to_model(_db_path(), season, week, req.game_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/week/{season}/{week}/lock")
def lock_week(season: int, week: int, req: LockRequest | None = None):
    try:
        source = req.source if req else "api"
        return create_weekly_submission(_db_path(), season, week, source=source)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/week/{season}/{week}/submission")
def get_submission(season: int, week: int):
    submission = get_weekly_submission(_db_path(), season, week)
    return {"submission": submission}


def _run_refresh(season: int, week: int) -> None:
    subprocess.run(
        [sys.executable, "scripts/refresh_weekly.py", "--season", str(season), "--week", str(week)],
        check=True,
    )


@app.post("/refresh/{season}/{week}")
def refresh(season: int, week: int, background_tasks: BackgroundTasks):
    """Trigger a weekly refresh in the background."""
    background_tasks.add_task(_run_refresh, season, week)
    return {"ok": True, "message": f"Refresh queued for season={season} week={week}"}
