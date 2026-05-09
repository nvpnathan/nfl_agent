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


from src.db.queries import insert_game_odds, get_game_odds


def _game(espn_id, season, week, home, away, home_score, away_score, home_win, game_date):
    return {
        "espn_id": espn_id,
        "season": season,
        "week": week,
        "game_type": "regular",
        "home_team": home,
        "away_team": away,
        "home_espn_id": "1",
        "away_espn_id": "2",
        "game_date": game_date,
        "venue": "X",
        "is_indoor": 0,
        "is_neutral": 0,
        "attendance": None,
        "home_score": home_score,
        "away_score": away_score,
        "home_win": home_win,
    }


def test_insert_and_get_game_odds(tmp_path):
    db_path = str(tmp_path / "test.db")
    create_schema(db_path)
    from src.db.queries import insert_espn_game
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
    from src.db.queries import insert_espn_game
    insert_espn_game(db_path, _game("g1", 2024, 1, "KC", "BAL", 27, 20, 1, "2024-09-05T20:00Z"))
    insert_game_odds(db_path, {"espn_id": "g1", "home_spread": -2.5, "game_total": 46.5,
                               "home_moneyline": "4/6", "away_moneyline": "5/4"})
    insert_game_odds(db_path, {"espn_id": "g1", "home_spread": -3.0, "game_total": 47.0,
                               "home_moneyline": "1/2", "away_moneyline": "7/4"})
    row = get_game_odds(db_path, "g1")
    assert row["home_spread"] == -3.0
