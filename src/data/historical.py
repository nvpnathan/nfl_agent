"""DB-backed game data access layer. All data sourced from ESPN via SQLite."""
import sqlite3
from typing import Optional
import pandas as pd
from src.db.queries import get_team_results

PLAYOFF_TYPES = {"wildcard", "divisional", "conference", "superbowl"}


def load_games(db_path: str, seasons: list[int]) -> pd.DataFrame:
    """Load completed games from DB for given seasons."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(seasons))
    rows = conn.execute(
        f"SELECT * FROM games WHERE season IN ({placeholders}) AND home_win IS NOT NULL"
        f" ORDER BY game_date",
        seasons,
    ).fetchall()
    conn.close()
    return pd.DataFrame([dict(r) for r in rows])


def load_schedules(db_path: str, seasons: list[int]) -> pd.DataFrame:
    """Load all games (including upcoming) from DB for given seasons."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(seasons))
    rows = conn.execute(
        f"SELECT * FROM games WHERE season IN ({placeholders}) ORDER BY game_date",
        seasons,
    ).fetchall()
    conn.close()
    return pd.DataFrame([dict(r) for r in rows])


def get_team_recent_form(db_path: str, team: str, season: int,
                          week: int, n: int = 4) -> dict:
    results = get_team_results(db_path, team, season, week, n)
    if not results:
        return {"win_pct": 0.5, "avg_point_diff": 0.0, "games_played": 0}
    wins = 0
    point_diffs = []
    for r in results:
        if r["home_team"] == team:
            wins += int(r["home_win"] == 1)
            point_diffs.append(r["home_score"] - r["away_score"])
        else:
            wins += int(r["home_win"] == 0)
            point_diffs.append(r["away_score"] - r["home_score"])
    return {
        "win_pct": wins / len(results),
        "avg_point_diff": sum(point_diffs) / len(point_diffs),
        "games_played": len(results),
    }


def get_home_road_winpct(db_path: str, team: str, season: int,
                          week: int, home_games: bool, n: int = 4) -> float:
    """Win% in home games only or road games only over last n such games."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if home_games:
        rows = conn.execute("""
            SELECT home_win FROM games
            WHERE home_team=? AND season=? AND week<? AND home_win IS NOT NULL
            ORDER BY game_date DESC LIMIT ?
        """, (team, season, week, n)).fetchall()
        wins = sum(1 for r in rows if r["home_win"] == 1)
    else:
        rows = conn.execute("""
            SELECT home_win FROM games
            WHERE away_team=? AND season=? AND week<? AND home_win IS NOT NULL
            ORDER BY game_date DESC LIMIT ?
        """, (team, season, week, n)).fetchall()
        wins = sum(1 for r in rows if r["home_win"] == 0)
    conn.close()
    return wins / len(rows) if rows else 0.5


def get_rest_days(db_path: str, team: str, season: int, week: int) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    prior = conn.execute("""
        SELECT game_date FROM games
        WHERE (home_team=? OR away_team=?) AND season=? AND week<?
          AND home_win IS NOT NULL
        ORDER BY game_date DESC LIMIT 1
    """, (team, team, season, week)).fetchone()
    current = conn.execute("""
        SELECT game_date FROM games
        WHERE (home_team=? OR away_team=?) AND season=? AND week=?
        LIMIT 1
    """, (team, team, season, week)).fetchone()
    conn.close()

    if not prior:
        return 14
    if not current:
        return 7
    last_date = pd.to_datetime(prior["game_date"])
    game_date = pd.to_datetime(current["game_date"])
    return max(1, (game_date - last_date).days)


def get_team_sos(db_path: str, team: str, season: int, week: int, n: int = 4) -> float:
    """Avg win% of recent opponents (strength of schedule)."""
    results = get_team_results(db_path, team, season, week, n)
    if not results:
        return 0.5
    opp_winpcts = []
    for r in results:
        opp = r["away_team"] if r["home_team"] == team else r["home_team"]
        opp_form = get_team_recent_form(db_path, opp, season, r["week"], n=4)
        opp_winpcts.append(opp_form["win_pct"])
    return sum(opp_winpcts) / len(opp_winpcts)
