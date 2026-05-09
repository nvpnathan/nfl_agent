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
