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
