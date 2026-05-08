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

@pytest.fixture
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    create_schema(db_path)
    return db_path

def test_insert_and_fetch_game(tmp_db):
    from src.db.queries import insert_game, get_games_for_week
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
        "home_score": None,
        "away_score": None,
        "home_win": None,
    }
    insert_game(tmp_db, game)
    games = get_games_for_week(tmp_db, 2024, 1)
    assert len(games) == 1
    assert games[0]["game_id"] == "2024_01_KC_BAL"
