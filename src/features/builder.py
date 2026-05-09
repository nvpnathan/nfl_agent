import logging
import pandas as pd
from typing import Optional
from src.db.queries import get_team_box_stats, get_injuries_for_week

FEATURE_COLS = [
    "odds_home_win_prob",
    "home_rest_days", "away_rest_days", "rest_advantage",
    "home_recent_winpct", "away_recent_winpct",
    "home_home_winpct", "away_road_winpct",
    "home_recent_point_diff", "away_recent_point_diff",
    "home_turnover_diff_4wk", "away_turnover_diff_4wk",
    "home_total_yards_4wk", "away_total_yards_4wk",
    "home_third_down_pct_4wk", "away_third_down_pct_4wk",
    "home_qb_active", "away_qb_active",
    "home_key_injuries", "away_key_injuries",
    "is_indoor", "is_neutral",
    "temperature", "wind_speed",
    "home_sos", "away_sos",
    "is_playoff",
]

PLAYOFF_TYPES = {"wildcard", "divisional", "conference", "superbowl"}
OUT_STATUSES = {"Out", "Doubtful", "IR", "PUP-R"}
KEY_POSITIONS = {"QB", "RB", "WR", "TE", "OT", "OG", "C", "DE", "DT", "LB", "CB", "S"}


def _box_features(db_path: str, team: str, season: int, week: int) -> dict:
    rows = get_team_box_stats(db_path, team, season, week, n=4)
    if not rows:
        return {"turnover_diff": 0.0, "total_yards": 0.0, "third_down_pct": 0.35}
    turnover_diffs, yards, third_pcts = [], [], []
    for r in rows:
        turnover_diffs.append(-(r.get("turnovers") or 0))
        yards.append(r.get("total_yards") or 0)
        att = r.get("third_down_att") or 1
        made = r.get("third_down_made") or 0
        third_pcts.append(made / att)
    n = len(rows)
    return {
        "turnover_diff": sum(turnover_diffs) / n,
        "total_yards": sum(yards) / n,
        "third_down_pct": sum(third_pcts) / n,
    }


def _injury_features(db_path: str, team: str, season: int, week: int) -> dict:
    injuries = get_injuries_for_week(db_path, team, season, week)
    qb_out = any(
        i["is_qb"] and i.get("status") in OUT_STATUSES for i in injuries
    )
    key_out = sum(
        1 for i in injuries
        if not i["is_qb"]
        and i.get("status") in OUT_STATUSES
        and i.get("position") in KEY_POSITIONS
    )
    return {"qb_active": 0 if qb_out else 1, "key_injuries": key_out}


def build_features_for_game(
    game: dict,
    db_path: str,
    odds_home_win_prob: float,
    weather: Optional[dict] = None,
) -> dict:
    from src.data.historical import (
        get_team_recent_form, get_rest_days,
        get_home_road_winpct, get_team_sos,
    )
    season = int(game["season"])
    week = int(game["week"])
    home = game["home_team"]
    away = game["away_team"]

    home_form = get_team_recent_form(db_path, home, season, week)
    away_form = get_team_recent_form(db_path, away, season, week)
    home_rest = get_rest_days(db_path, home, season, week)
    away_rest = get_rest_days(db_path, away, season, week)
    home_home_pct = get_home_road_winpct(db_path, home, season, week, home_games=True)
    away_road_pct = get_home_road_winpct(db_path, away, season, week, home_games=False)
    home_sos = get_team_sos(db_path, home, season, week)
    away_sos = get_team_sos(db_path, away, season, week)

    home_box = _box_features(db_path, home, season, week)
    away_box = _box_features(db_path, away, season, week)
    home_inj = _injury_features(db_path, home, season, week)
    away_inj = _injury_features(db_path, away, season, week)

    is_indoor = int(game.get("is_indoor", 0))
    if is_indoor:
        temperature, wind_speed = 68.0, 0.0
    else:
        temperature = weather.get("temperature", 65.0) if weather else 65.0
        wind_speed = weather.get("wind_speed", 5.0) if weather else 5.0

    return {
        "odds_home_win_prob": odds_home_win_prob,
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "rest_advantage": home_rest - away_rest,
        "home_recent_winpct": home_form["win_pct"],
        "away_recent_winpct": away_form["win_pct"],
        "home_home_winpct": home_home_pct,
        "away_road_winpct": away_road_pct,
        "home_recent_point_diff": home_form["avg_point_diff"],
        "away_recent_point_diff": away_form["avg_point_diff"],
        "home_turnover_diff_4wk": home_box["turnover_diff"],
        "away_turnover_diff_4wk": away_box["turnover_diff"],
        "home_total_yards_4wk": home_box["total_yards"],
        "away_total_yards_4wk": away_box["total_yards"],
        "home_third_down_pct_4wk": home_box["third_down_pct"],
        "away_third_down_pct_4wk": away_box["third_down_pct"],
        "home_qb_active": home_inj["qb_active"],
        "away_qb_active": away_inj["qb_active"],
        "home_key_injuries": home_inj["key_injuries"],
        "away_key_injuries": away_inj["key_injuries"],
        "is_indoor": is_indoor,
        "is_neutral": int(game.get("is_neutral", 0)),
        "temperature": temperature,
        "wind_speed": wind_speed,
        "home_sos": home_sos,
        "away_sos": away_sos,
        "is_playoff": int(game.get("game_type", "regular") in PLAYOFF_TYPES),
    }


def build_training_dataset(
    db_path: str,
    seasons: list[int],
    odds_by_game: Optional[dict] = None,
) -> pd.DataFrame:
    from src.data.historical import load_games
    games = load_games(db_path, seasons)
    rows = []
    for _, game in games.iterrows():
        g = dict(game)
        odds_prob = (odds_by_game or {}).get(g["espn_id"], 0.55)
        try:
            features = build_features_for_game(g, db_path, odds_prob)
            features["home_win"] = int(g["home_win"])
            features["espn_id"] = g["espn_id"]
            features["season"] = int(g["season"])
            features["week"] = int(g["week"])
            features["home_team"] = g["home_team"]
            features["away_team"] = g["away_team"]
            rows.append(features)
        except Exception as e:
            logging.warning("Skipping game %s: %s", g.get("espn_id"), e)
    return pd.DataFrame(rows)
