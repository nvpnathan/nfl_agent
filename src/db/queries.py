import sqlite3
from typing import Optional


def _conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def get_existing_espn_ids(db_path: str) -> set:
    with _conn(db_path) as conn:
        rows = conn.execute("SELECT espn_id FROM games").fetchall()
    return {r[0] for r in rows}


def insert_espn_game(db_path: str, game: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO games
            (espn_id, season, week, game_type, home_team, away_team,
             home_espn_id, away_espn_id, game_date, venue,
             is_indoor, is_neutral, attendance, home_score, away_score, home_win)
            VALUES (:espn_id, :season, :week, :game_type, :home_team, :away_team,
                    :home_espn_id, :away_espn_id, :game_date, :venue,
                    :is_indoor, :is_neutral, :attendance,
                    :home_score, :away_score, :home_win)
        """, game)


def insert_team_stats(db_path: str, stats: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO team_game_stats
            (espn_id, team, is_home, total_yards, pass_yards, rush_yards,
             turnovers, first_downs, third_down_att, third_down_made,
             red_zone_att, red_zone_made, possession_secs, sacks_taken)
            VALUES (:espn_id, :team, :is_home, :total_yards, :pass_yards, :rush_yards,
                    :turnovers, :first_downs, :third_down_att, :third_down_made,
                    :red_zone_att, :red_zone_made, :possession_secs, :sacks_taken)
        """, stats)


def insert_injury(db_path: str, injury: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT OR IGNORE INTO injury_reports
            (season, week, team, athlete_id, athlete_name, position, status, is_qb)
            VALUES (:season, :week, :team, :athlete_id, :athlete_name,
                    :position, :status, :is_qb)
        """, injury)


def insert_depth_chart_entry(db_path: str, entry: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO depth_charts
            (season, week, team, athlete_id, athlete_name, rank)
            VALUES (:season, :week, :team, :athlete_id, :athlete_name, :rank)
        """, entry)


def insert_game_odds(db_path: str, odds: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO game_odds
            (espn_id, home_spread, game_total, home_moneyline, away_moneyline)
            VALUES (:espn_id, :home_spread, :game_total, :home_moneyline, :away_moneyline)
        """, odds)


def get_game_odds(db_path: str, espn_id: str) -> dict | None:
    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM game_odds WHERE espn_id = ?", (espn_id,)
        ).fetchone()
    return dict(row) if row else None


def get_games_for_week(db_path: str, season: int, week: int) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM games WHERE season=? AND week=?", (season, week)
        ).fetchall()
    return [dict(r) for r in rows]


def get_games_for_season(db_path: str, season: int) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM games WHERE season=? ORDER BY game_date", (season,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_team_results(db_path: str, team: str, season: int,
                     before_week: int, n: int = 4) -> list[dict]:
    """Last n completed games for team before before_week. Most recent first."""
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT espn_id, season, week, game_date, home_team, away_team,
                   home_score, away_score, home_win
            FROM games
            WHERE (home_team=? OR away_team=?)
              AND season=? AND week<? AND home_win IS NOT NULL
            ORDER BY game_date DESC
            LIMIT ?
        """, (team, team, season, before_week, n)).fetchall()
    return [dict(r) for r in rows]


def get_team_box_stats(db_path: str, team: str, season: int,
                       before_week: int, n: int = 4) -> list[dict]:
    """Last n box score rows for team before before_week. Most recent first."""
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT t.*
            FROM team_game_stats t
            JOIN games g USING(espn_id)
            WHERE t.team=? AND g.season=? AND g.week<? AND g.home_win IS NOT NULL
            ORDER BY g.game_date DESC
            LIMIT ?
        """, (team, season, before_week, n)).fetchall()
    return [dict(r) for r in rows]


def get_injuries_for_week(db_path: str, team: str, season: int, week: int) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT * FROM injury_reports WHERE team=? AND season=? AND week=?
        """, (team, season, week)).fetchall()
    return [dict(r) for r in rows]


def get_starting_qb(db_path: str, team: str, season: int, week: int) -> Optional[dict]:
    with _conn(db_path) as conn:
        row = conn.execute("""
            SELECT * FROM depth_charts
            WHERE team=? AND season=? AND week=? AND rank=1
        """, (team, season, week)).fetchone()
    return dict(row) if row else None


def insert_prediction(db_path: str, pred: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT INTO predictions
            (game_id, season, week, home_win_prob, odds_implied_prob,
             model_version, predicted_winner)
            VALUES (:game_id, :season, :week, :home_win_prob, :odds_implied_prob,
                    :model_version, :predicted_winner)
        """, pred)


def get_predictions_for_week(db_path: str, season: int, week: int) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT p.*, g.home_team, g.away_team, g.game_date, g.is_indoor
            FROM predictions p JOIN games g ON p.game_id = g.espn_id
            WHERE p.season=? AND p.week=?
            ORDER BY p.home_win_prob DESC
        """, (season, week)).fetchall()
    return [dict(r) for r in rows]


def upsert_weekly_assignment(db_path: str, assignment: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT INTO weekly_assignments
            (season, week, game_id, predicted_winner, confidence_points,
             win_probability, is_uncertain, is_overridden, override_reason, updated_at)
            VALUES (:season, :week, :game_id, :predicted_winner, :confidence_points,
                    :win_probability, :is_uncertain, :is_overridden, :override_reason,
                    datetime('now'))
            ON CONFLICT(season, week, game_id) DO UPDATE SET
                confidence_points=excluded.confidence_points,
                predicted_winner=excluded.predicted_winner,
                win_probability=excluded.win_probability,
                is_uncertain=excluded.is_uncertain,
                is_overridden=excluded.is_overridden,
                override_reason=excluded.override_reason,
                updated_at=datetime('now')
        """, {**assignment, "is_overridden": assignment.get("is_overridden", 0),
               "override_reason": assignment.get("override_reason")})


def get_weekly_assignments(db_path: str, season: int, week: int) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT wa.*, g.home_team, g.away_team, g.game_date
            FROM weekly_assignments wa JOIN games g ON wa.game_id = g.espn_id
            WHERE wa.season=? AND wa.week=?
            ORDER BY wa.confidence_points DESC
        """, (season, week)).fetchall()
    return [dict(r) for r in rows]


def insert_reranking(db_path: str, reranking: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT INTO rerankings
            (session_id, season, week, game_id, old_points, new_points, reason)
            VALUES (:session_id, :season, :week, :game_id, :old_points, :new_points, :reason)
        """, reranking)


def get_rerankings_for_session(db_path: str, session_id: str) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM rerankings WHERE session_id=? ORDER BY created_at",
            (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def insert_conversation_message(db_path: str, message: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT INTO conversations (session_id, role, content, season, week)
            VALUES (:session_id, :role, :content, :season, :week)
        """, message)


def get_conversation_history(db_path: str, session_id: str) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE session_id=? ORDER BY created_at",
            (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]
