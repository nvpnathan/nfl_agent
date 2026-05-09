import os
import pytest
os.environ["NFL_DB_PATH"] = ":memory:"

from fastapi.testclient import TestClient
from src.db.schema import create_schema
from src.db.queries import insert_espn_game, upsert_weekly_assignment
from src.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _seed(tmp_path, monkeypatch):
    db = str(tmp_path / "test.db")
    monkeypatch.setenv("NFL_DB_PATH", db)
    create_schema(db)
    insert_espn_game(db, {
        "espn_id": "g1", "season": 2024, "week": 10, "game_type": "regular",
        "home_team": "BAL", "away_team": "CIN", "home_espn_id": "1", "away_espn_id": "2",
        "game_date": "2024-11-07T20:00Z", "venue": "M&T Bank Stadium",
        "is_indoor": 0, "is_neutral": 0, "attendance": None,
        "home_score": None, "away_score": None, "home_win": None,
    })
    upsert_weekly_assignment(db, {
        "season": 2024, "week": 10, "game_id": "g1",
        "predicted_winner": "BAL", "confidence_points": 14,
        "win_probability": 0.84, "is_uncertain": 0,
        "is_overridden": 0, "override_reason": None,
    })


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_get_week_returns_assignments():
    resp = client.get("/week/2024/10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["season"] == 2024
    assert data["week"] == 10
    assert len(data["assignments"]) == 1
    assert data["assignments"][0]["confidence_points"] == 14


def test_override_pick():
    resp = client.post("/week/2024/10/override", json={
        "game_id": "g1", "confidence_points": 14, "reason": "gut feel"
    })
    assert resp.status_code == 200
    assert resp.json()["confidence_points"] == 14


def test_override_nonexistent_game():
    resp = client.post("/week/2024/10/override", json={
        "game_id": "bad_id", "confidence_points": 5
    })
    assert resp.status_code == 404


def test_swap_enforces_permutation(tmp_path, monkeypatch):
    """Swapping g1 (14pts) to 10pts should give g2 14pts."""
    from src.db.queries import insert_espn_game, upsert_weekly_assignment, get_weekly_assignments, swap_confidence_points
    from src.db.schema import create_schema
    db = str(tmp_path / "swap.db")
    monkeypatch.setenv("NFL_DB_PATH", db)
    create_schema(db)

    for espn_id, home, away, pts in [("g1", "BAL", "CIN", 14), ("g2", "KC", "DEN", 10)]:
        insert_espn_game(db, {
            "espn_id": espn_id, "season": 2024, "week": 10, "game_type": "regular",
            "home_team": home, "away_team": away, "home_espn_id": "1", "away_espn_id": "2",
            "game_date": "2024-11-07T20:00Z", "venue": "Stadium",
            "is_indoor": 0, "is_neutral": 0, "attendance": None,
            "home_score": None, "away_score": None, "home_win": None,
        })
        upsert_weekly_assignment(db, {
            "season": 2024, "week": 10, "game_id": espn_id,
            "predicted_winner": home, "confidence_points": pts,
            "win_probability": 0.7, "is_uncertain": 0,
            "is_overridden": 0, "override_reason": None,
        })

    swap_confidence_points(db, 2024, 10, "g1", 10, "test swap")

    result = {a["game_id"]: a["confidence_points"] for a in get_weekly_assignments(db, 2024, 10)}
    assert result["g1"] == 10
    assert result["g2"] == 14  # displaced game gets old points
    # Confirm it's still a valid permutation (no duplicates)
    assert sorted(result.values()) == [10, 14]
