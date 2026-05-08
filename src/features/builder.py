import logging
import pandas as pd
import numpy as np
from typing import Optional
from src.data.historical import load_games, load_schedules, get_team_recent_form, get_rest_days

FEATURE_COLS = [
    "odds_home_win_prob", "home_rest_days", "away_rest_days", "rest_advantage",
    "home_qb_out", "away_qb_out", "home_recent_winpct", "away_recent_winpct",
    "home_recent_point_diff", "away_recent_point_diff",
    "temperature", "wind_speed", "home_sos", "away_sos", "is_playoff",
]

PLAYOFF_TYPES = {"wildcard", "divisional", "conference", "superbowl"}


def _get_sos(schedules: pd.DataFrame, team: str, season: int, week: int,
             n: int = 4) -> float:
    mask = (
        ((schedules["home_team"] == team) | (schedules["away_team"] == team)) &
        (schedules["season"] == season) & (schedules["week"] < week) &
        (schedules["home_score"].notna())
    )
    recent = schedules[mask].tail(n)
    if recent.empty:
        return 0.5
    opponent_winpcts = []
    for _, row in recent.iterrows():
        opp = row["away_team"] if row["home_team"] == team else row["home_team"]
        opp_form = get_team_recent_form(schedules, opp, season, row["week"], n=4)
        opponent_winpcts.append(opp_form["win_pct"])
    return float(np.mean(opponent_winpcts))


def build_features_for_game(
    game_row: pd.Series,
    schedules: pd.DataFrame,
    odds_home_win_prob: float,
    home_qb_out: bool = False,
    away_qb_out: bool = False,
    weather: Optional[dict] = None,
) -> dict:
    season = int(game_row["season"])
    week = int(game_row["week"])
    home = game_row["home_team"]
    away = game_row["away_team"]

    home_rest = get_rest_days(schedules, home, season, week)
    away_rest = get_rest_days(schedules, away, season, week)
    home_form = get_team_recent_form(schedules, home, season, week)
    away_form = get_team_recent_form(schedules, away, season, week)
    home_sos = _get_sos(schedules, home, season, week)
    away_sos = _get_sos(schedules, away, season, week)

    temperature = weather.get("temperature") if weather else None
    wind_speed = weather.get("wind_speed") if weather else None

    return {
        "odds_home_win_prob": odds_home_win_prob,
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "rest_advantage": home_rest - away_rest,
        "home_qb_out": int(home_qb_out),
        "away_qb_out": int(away_qb_out),
        "home_recent_winpct": home_form["win_pct"],
        "away_recent_winpct": away_form["win_pct"],
        "home_recent_point_diff": home_form["avg_point_diff"],
        "away_recent_point_diff": away_form["avg_point_diff"],
        "temperature": temperature if temperature is not None else 65.0,
        "wind_speed": wind_speed if wind_speed is not None else 5.0,
        "home_sos": home_sos,
        "away_sos": away_sos,
        "is_playoff": int(game_row.get("game_type", "regular") in PLAYOFF_TYPES),
    }


def build_training_dataset(
    seasons: list[int],
    odds_by_game: Optional[dict] = None,
) -> pd.DataFrame:
    schedules = load_schedules(seasons)
    games = load_games(seasons)
    rows = []
    for _, game in games.iterrows():
        odds_prob = (odds_by_game or {}).get(game["game_id"], 0.55)
        try:
            features = build_features_for_game(
                game_row=game,
                schedules=schedules,
                odds_home_win_prob=odds_prob,
                home_qb_out=False,
                away_qb_out=False,
                weather=None,
            )
            features["home_win"] = int(game["home_win"])
            features["game_id"] = game["game_id"]
            features["season"] = int(game["season"])
            features["week"] = int(game["week"])
            rows.append(features)
        except Exception as e:
            logging.warning("Skipping game %s: %s", game.get("game_id", "unknown"), e)
            continue
    return pd.DataFrame(rows)
