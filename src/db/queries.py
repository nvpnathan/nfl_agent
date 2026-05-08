import sqlite3
from typing import Optional

def _conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def insert_game(db_path: str, game: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO games
            (game_id, season, week, game_type, home_team, away_team,
             game_date, stadium, is_outdoor, home_score, away_score, home_win)
            VALUES (:game_id, :season, :week, :game_type, :home_team, :away_team,
                    :game_date, :stadium, :is_outdoor,
                    :home_score, :away_score, :home_win)
        """, {**game, "home_score": game.get("home_score"),
               "away_score": game.get("away_score"), "home_win": game.get("home_win")})

def get_games_for_week(db_path: str, season: int, week: int) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM games WHERE season=? AND week=?", (season, week)
        ).fetchall()
    return [dict(r) for r in rows]

def get_games_for_season(db_path: str, season: int) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM games WHERE season=? ORDER BY week", (season,)
        ).fetchall()
    return [dict(r) for r in rows]

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
            SELECT p.*, g.home_team, g.away_team, g.game_date, g.is_outdoor
            FROM predictions p JOIN games g USING(game_id)
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
            FROM weekly_assignments wa JOIN games g USING(game_id)
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

def upsert_injury_report(db_path: str, injury: dict) -> None:
    with _conn(db_path) as conn:
        conn.execute("""
            INSERT INTO injury_reports
            (team, player_name, position, injury_status, is_qb, season, week)
            VALUES (:team, :player_name, :position, :injury_status, :is_qb, :season, :week)
        """, injury)

def get_injuries_for_week(db_path: str, season: int, week: int,
                           team: Optional[str] = None) -> list[dict]:
    with _conn(db_path) as conn:
        if team:
            rows = conn.execute("""
                SELECT * FROM injury_reports
                WHERE season=? AND week=? AND team=?
                ORDER BY is_qb DESC
            """, (season, week, team)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM injury_reports WHERE season=? AND week=?
                ORDER BY team, is_qb DESC
            """, (season, week)).fetchall()
    return [dict(r) for r in rows]
