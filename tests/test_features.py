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
