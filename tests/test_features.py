import pytest
from src.data.historical import load_games, load_schedules

def test_load_schedules_returns_dataframe():
    df = load_schedules([2023])
    assert len(df) > 200
    required_cols = {"game_id", "season", "week", "home_team", "away_team",
                     "home_score", "away_score", "game_type", "gameday"}
    assert required_cols.issubset(set(df.columns))

def test_load_games_marks_home_win():
    df = load_games([2023])
    assert "home_win" in df.columns
    assert df["home_win"].isin([0, 1]).all()
    assert df["home_win"].notna().all()

from src.features.builder import build_features_for_game, build_training_dataset

def test_build_features_for_game_has_all_keys():
    schedules_df = load_schedules([2023])
    game_row = schedules_df[schedules_df["week"] == 5].iloc[0]
    features = build_features_for_game(
        game_row=game_row,
        schedules=schedules_df,
        odds_home_win_prob=0.65,
        home_qb_out=False,
        away_qb_out=False,
        weather={"temperature": 55, "wind_speed": 8, "is_outdoor": True},
    )
    required = {
        "odds_home_win_prob", "home_rest_days", "away_rest_days",
        "rest_advantage", "home_qb_out", "away_qb_out",
        "home_recent_winpct", "away_recent_winpct",
        "home_recent_point_diff", "away_recent_point_diff",
        "temperature", "wind_speed", "home_sos", "away_sos", "is_playoff",
    }
    assert required.issubset(set(features.keys()))

def test_build_training_dataset_has_correct_shape():
    df = build_training_dataset(seasons=[2022])
    assert len(df) > 200
    assert "home_win" in df.columns
    assert df["home_win"].isin([0, 1]).all()
