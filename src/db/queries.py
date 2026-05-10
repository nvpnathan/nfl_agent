import sqlite3
from typing import Optional


def _conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def _tier(prob: float) -> str:
    if prob >= 0.70:
        return "lock"
    if prob >= 0.62:
        return "lean"
    return "toss"


def _market_text(row: dict) -> str | None:
    spread = row.get("home_spread")
    total = row.get("game_total")
    if spread is None or total is None:
        return None
    if spread < -0.1:
        fav = f"{row['home_team']} {spread:.1f}"
    elif spread > 0.1:
        fav = f"{row['away_team']} -{spread:.1f}"
    else:
        fav = "EVEN"
    return f"{fav}  ·  O/U {total:.1f}"


def _model_point_map(assignments: list[dict]) -> dict[str, int]:
    model_order = sorted(assignments, key=lambda a: a["win_probability"], reverse=True)
    return {
        a["game_id"]: len(model_order) - i
        for i, a in enumerate(model_order)
    }


def _normalize_weekly_overrides(conn: sqlite3.Connection,
                                season: int, week: int) -> None:
    rows = conn.execute("""
        SELECT game_id, confidence_points, win_probability
        FROM weekly_assignments
        WHERE season=? AND week=?
    """, (season, week)).fetchall()
    assignments = [dict(r) for r in rows]
    model_points = _model_point_map(assignments)

    for assignment in assignments:
        is_overridden = int(
            assignment["confidence_points"] != model_points[assignment["game_id"]]
        )
        if is_overridden:
            conn.execute("""
                UPDATE weekly_assignments
                SET is_overridden=1, updated_at=datetime('now')
                WHERE season=? AND week=? AND game_id=?
            """, (season, week, assignment["game_id"]))
        else:
            conn.execute("""
                UPDATE weekly_assignments
                SET is_overridden=0, override_reason=NULL, updated_at=datetime('now')
                WHERE season=? AND week=? AND game_id=?
            """, (season, week, assignment["game_id"]))


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


def swap_confidence_points(db_path: str, season: int, week: int,
                           game_id: str, new_points: int,
                           reason: str | None = None) -> tuple[str | None, int | None]:
    """Set game_id to new_points, swapping with whoever currently holds that value.

    Returns (displaced_game_id, old_points_of_target) so callers can report what changed.
    No-op if game_id already has new_points.
    """
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT game_id, confidence_points, predicted_winner, win_probability,
                   is_uncertain, is_overridden, override_reason
            FROM weekly_assignments
            WHERE season=? AND week=?
        """, (season, week)).fetchall()

    current = {r["game_id"]: dict(r) for r in rows}
    target = current.get(game_id)
    if target is None or target["confidence_points"] == new_points:
        return None, None

    old_points = target["confidence_points"]
    displaced = next(
        (a for gid, a in current.items() if a["confidence_points"] == new_points and gid != game_id),
        None,
    )

    with _conn(db_path) as conn:
        # Give displaced game the old points and its original prediction back
        if displaced:
            conn.execute("""
                UPDATE weekly_assignments
                SET confidence_points=?, predicted_winner=?, is_overridden=1, updated_at=datetime('now')
                WHERE season=? AND week=? AND game_id=?
            """, (old_points, displaced["predicted_winner"], season, week, displaced["game_id"]))

        # Update target game with new points and the displaced game's prediction
        conn.execute("""
            UPDATE weekly_assignments
            SET confidence_points=?, predicted_winner=?, is_overridden=1, override_reason=?, updated_at=datetime('now')
            WHERE season=? AND week=? AND game_id=?
        """, (new_points, displaced["predicted_winner"] if displaced else target["predicted_winner"],
              reason, season, week, game_id))

    return displaced["game_id"] if displaced else None, old_points


def revert_assignment_to_model(db_path: str, season: int, week: int,
                               game_id: str) -> dict:
    """Move one game back to its model-assigned confidence points."""
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT game_id, confidence_points, predicted_winner, win_probability,
                   is_uncertain, is_overridden, override_reason
            FROM weekly_assignments
            WHERE season=? AND week=?
        """, (season, week)).fetchall()

        current = {r["game_id"]: dict(r) for r in rows}
        target = current.get(game_id)
        if target is None:
            raise ValueError("Assignment not found")

        model_points = _model_point_map(list(current.values()))
        target_model_points = model_points[game_id]
        old_points = target["confidence_points"]
        displaced = next(
            (
                a for gid, a in current.items()
                if a["confidence_points"] == target_model_points and gid != game_id
            ),
            None,
        )

        if old_points != target_model_points:
            if displaced:
                conn.execute("""
                    UPDATE weekly_assignments
                    SET confidence_points=?, is_overridden=1, updated_at=datetime('now')
                    WHERE season=? AND week=? AND game_id=?
                """, (old_points, season, week, displaced["game_id"]))

            conn.execute("""
                UPDATE weekly_assignments
                SET confidence_points=?, is_overridden=0, override_reason=NULL,
                    updated_at=datetime('now')
                WHERE season=? AND week=? AND game_id=?
            """, (target_model_points, season, week, game_id))

        _normalize_weekly_overrides(conn, season, week)

    return {
        "ok": True,
        "game_id": game_id,
        "old_points": old_points,
        "model_points": target_model_points,
        "displaced_game_id": displaced["game_id"] if displaced else None,
    }


def get_weekly_assignments(db_path: str, season: int, week: int) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT wa.*, g.home_team, g.away_team, g.game_date
            FROM weekly_assignments wa JOIN games g ON wa.game_id = g.espn_id
            WHERE wa.season=? AND wa.week=?
            ORDER BY wa.confidence_points DESC
        """, (season, week)).fetchall()
    return [dict(r) for r in rows]


def create_weekly_submission(db_path: str, season: int, week: int,
                             source: str = "streamlit") -> dict:
    """Persist the current weekly picks as the locked submission snapshot.

    The model points are reconstructed from the model win probabilities, matching
    the confidence optimizer's ordering. The submitted points are the current
    possibly overridden values in weekly_assignments.
    """
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT wa.*, g.home_team, g.away_team,
                   go.home_spread, go.game_total
            FROM weekly_assignments wa
            JOIN games g ON wa.game_id = g.espn_id
            LEFT JOIN game_odds go ON wa.game_id = go.espn_id
            WHERE wa.season=? AND wa.week=?
        """, (season, week)).fetchall()

        assignments = [dict(r) for r in rows]
        if not assignments:
            raise ValueError(f"No assignments for season={season} week={week}")

        model_order = sorted(
            assignments, key=lambda a: a["win_probability"], reverse=True
        )
        model_points = {
            a["game_id"]: len(model_order) - i
            for i, a in enumerate(model_order)
        }

        model_picks = {r["game_id"]: r["predicted_winner"] for r in conn.execute("""
            SELECT game_id, predicted_winner FROM predictions
            WHERE season=? AND week=?
        """, (season, week)).fetchall()}

        conn.execute("""
            INSERT INTO weekly_submissions
                (season, week, status, source, submitted_at)
            VALUES (?, ?, 'locked', ?, datetime('now'))
            ON CONFLICT(season, week) DO UPDATE SET
                status='locked',
                source=excluded.source,
                submitted_at=datetime('now')
        """, (season, week, source))

        submission = conn.execute("""
            SELECT * FROM weekly_submissions WHERE season=? AND week=?
        """, (season, week)).fetchone()
        submission_id = submission["submission_id"]

        conn.execute(
            "DELETE FROM weekly_submission_picks WHERE submission_id=?",
            (submission_id,),
        )

        for a in assignments:
            model_pts = model_points[a["game_id"]]
            submitted_pts = a["confidence_points"]
            model_pick = model_picks.get(a["game_id"], a["predicted_winner"])
            conn.execute("""
                INSERT INTO weekly_submission_picks
                    (submission_id, game_id, model_pick, submitted_pick,
                     model_points, submitted_points, points_delta,
                     win_probability, tier, market, is_overridden,
                     override_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                submission_id,
                a["game_id"],
                model_pick,
                a["predicted_winner"],
                model_pts,
                submitted_pts,
                submitted_pts - model_pts,
                a["win_probability"],
                _tier(a["win_probability"]),
                _market_text(a),
                int(a.get("is_overridden") or 0),
                a.get("override_reason"),
            ))

    submission_data = get_weekly_submission(db_path, season, week)
    if submission_data is None:
        raise RuntimeError("Submission save failed")
    return submission_data


def get_weekly_submission(db_path: str, season: int, week: int) -> dict | None:
    with _conn(db_path) as conn:
        submission = conn.execute("""
            SELECT * FROM weekly_submissions WHERE season=? AND week=?
        """, (season, week)).fetchone()
        if submission is None:
            return None
        picks = conn.execute("""
            SELECT sp.*, g.home_team, g.away_team
            FROM weekly_submission_picks sp
            JOIN games g ON sp.game_id = g.espn_id
            WHERE sp.submission_id=?
            ORDER BY sp.submitted_points DESC
        """, (submission["submission_id"],)).fetchall()

    result = dict(submission)
    result["picks"] = [dict(p) for p in picks]
    return result


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


def insert_model_training_run(db_path: str, run: dict) -> int:
    """Insert a training run record. Returns the new model_run_id."""
    with _conn(db_path) as conn:
        cursor = conn.execute("""
            INSERT INTO model_training_runs
                (model_version, cv_accuracy_mean, cv_accuracy_std, n_samples, seasons_used)
            VALUES (?, ?, ?, ?, ?)
        """, (run["model_version"], run["cv_accuracy_mean"], run["cv_accuracy_std"],
              run["n_samples"], run["seasons_used"]))
        return cursor.lastrowid


def get_model_training_runs(db_path: str) -> list[dict]:
    """Get all training runs, ordered by created_at descending."""
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT * FROM model_training_runs ORDER BY created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_seasons_with_submissions(db_path: str) -> list[int]:
    """Get all seasons that have at least one weekly submission."""
    with _conn(db_path) as conn:
        rows = conn.execute("""
            SELECT DISTINCT season FROM weekly_submissions ORDER BY season DESC
        """).fetchall()
    return [int(r[0]) for r in rows]


def get_weekly_analytics(db_path: str, season: int, week: int) -> dict | None:
    """Get analytics data for a specific season/week.

    Returns dict with keys: 'picks', 'overrides'.
    Each pick has: game_id, home_team, away_team, picked_team, model_picked_team,
    is_overridden, home_win, model_confidence, submitted_points, pick_correct (0/1),
    model_correct (0/1). Returns None if no submission exists.
    """
    with _conn(db_path) as conn:

        sub = conn.execute("""
            SELECT submission_id FROM weekly_submissions WHERE season=? AND week=?
        """, (season, week)).fetchone()
        if sub is None:
            return None

        submission_id = sub["submission_id"]

        picks_rows = conn.execute("""
            SELECT
                sp.game_id,
                g.home_team,
                g.away_team,
                CASE WHEN sp.submitted_pick = g.home_team THEN 'home' ELSE 'away' END AS pick_side,
                sp.submitted_pick AS picked_team,
                p.predicted_winner AS model_picked_team,
                sp.is_overridden,
                g.home_win,
                p.home_win_prob AS model_confidence,
                sp.submitted_points,
                CASE WHEN sp.submitted_pick = g.home_team AND g.home_win = 1 THEN 1
                     WHEN sp.submitted_pick = g.away_team AND g.home_win = 0 THEN 1
                     ELSE 0 END AS pick_correct,
                CASE WHEN p.predicted_winner = g.home_team AND g.home_win = 1 THEN 1
                     WHEN p.predicted_winner = g.away_team AND g.home_win = 0 THEN 1
                     ELSE 0 END AS model_correct
            FROM weekly_submission_picks sp
            JOIN games g ON sp.game_id = g.espn_id
            LEFT JOIN predictions p ON sp.game_id = p.game_id
                AND p.season=? AND p.week=?
            WHERE sp.submission_id=?
            ORDER BY sp.submitted_points DESC, g.game_date ASC
        """, (season, week, submission_id)).fetchall()

    picks = [dict(r) for r in picks_rows]
    if not picks:
        return None

    # Add source tag
    for p in picks:
        p["source"] = "override" if p["is_overridden"] else "model"

    overrides = [p for p in picks if p["is_overridden"]]

    return {"picks": picks, "overrides": overrides}


def get_season_override_summary(db_path: str, season: int) -> list[dict]:
    """Get override summary per week for a season.

    Returns list of dicts with: 'week', 'overrides_made', 'saved', 'hurt'.
    """
    with _conn(db_path) as conn:
        weeks = [int(r[0]) for r in conn.execute("""
            SELECT DISTINCT week FROM weekly_submissions WHERE season=? ORDER BY week
        """, (season,)).fetchall()]

    results = []
    for week in weeks:
        with _conn(db_path) as conn:
            sub = conn.execute("""
                SELECT submission_id FROM weekly_submissions WHERE season=? AND week=?
            """, (season, week)).fetchone()
        if sub is None:
            continue

        rows = conn.execute("""
            SELECT
                CASE WHEN sp.submitted_pick = g.home_team AND g.home_win = 1 THEN 1
                     WHEN sp.submitted_pick = g.away_team AND g.home_win = 0 THEN 1
                     ELSE 0 END AS user_correct,
                CASE WHEN p.predicted_winner = g.home_team AND g.home_win = 1 THEN 1
                     WHEN p.predicted_winner = g.away_team AND g.home_win = 0 THEN 1
                     ELSE 0 END AS model_correct
            FROM weekly_submission_picks sp
            JOIN games g ON sp.game_id = g.espn_id
            LEFT JOIN predictions p ON sp.game_id = p.game_id
                AND p.season=? AND p.week=?
            WHERE sp.submission_id=? AND sp.is_overridden=1
        """, (season, week, sub["submission_id"])).fetchall()

        total = len(rows)
        if total == 0:
            results.append({"week": week, "overrides_made": 0, "saved": 0, "hurt": 0})
            continue

        saved = sum(1 for r in rows if int(r["user_correct"]) == 1 and int(r["model_correct"]) == 0)
        hurt = sum(1 for r in rows if int(r["user_correct"]) == 0 and int(r["model_correct"]) == 1)

        results.append({
            "week": week,
            "overrides_made": total,
            "saved": saved,
            "hurt": hurt,
        })

    return results


def get_weekly_overall_stats(db_path: str, season: int) -> dict | None:
    """Get overall stats for a season across all weeks.

    Returns dict with keys: 'weeks_covered', 'actual_pct', 'model_pct',
    'overrides_total', 'saved', 'hurt'. Returns None if no data.
    """
    with _conn(db_path) as conn:
        weeks = [int(r[0]) for r in conn.execute("""
            SELECT DISTINCT week FROM weekly_submissions WHERE season=? ORDER BY week
        """, (season,)).fetchall()]

    if not weeks:
        return None

    rows = conn.execute("""
        SELECT
            sp.submitted_pick,
            g.home_team,
            g.away_team,
            g.home_win,
            p.predicted_winner AS model_picked,
            sp.is_overridden
        FROM weekly_submission_picks sp
        JOIN games g ON sp.game_id = g.espn_id
        JOIN weekly_submissions ws ON sp.submission_id = ws.submission_id
        LEFT JOIN predictions p ON sp.game_id = p.game_id
            AND p.season = ws.season AND p.week = ws.week
        WHERE ws.season=?
    """, (season,)).fetchall()

    total = len(rows)
    if total == 0:
        return None

    actual_correct = sum(
        1 for r in rows
        if (r["submitted_pick"] == r["home_team"] and r["home_win"] == 1)
        or (r["submitted_pick"] == r["away_team"] and r["home_win"] == 0)
    )
    model_correct = sum(
        1 for r in rows
        if (r["model_picked"] == r["home_team"] and r["home_win"] == 1)
        or (r["model_picked"] == r["away_team"] and r["home_win"] == 0)
    )

    override_rows = [r for r in rows if int(r["is_overridden"]) == 1]
    om = len(override_rows)
    saved = sum(
        1 for r in override_rows
        if ((r["submitted_pick"] == r["home_team"] and r["home_win"] == 1)
            or (r["submitted_pick"] == r["away_team"] and r["home_win"] == 0))
        and not ((r["model_picked"] == r["home_team"] and r["home_win"] == 1)
                 or (r["model_picked"] == r["away_team"] and r["home_win"] == 0))
    )
    hurt = sum(
        1 for r in override_rows
        if not ((r["submitted_pick"] == r["home_team"] and r["home_win"] == 1)
                or (r["submitted_pick"] == r["away_team"] and r["home_win"] == 0))
        and ((r["model_picked"] == r["home_team"] and r["home_win"] == 1)
             or (r["model_picked"] == r["away_team"] and r["home_win"] == 0))
    )

    return {
        "weeks_covered": len(weeks),
        "actual_pct": round(actual_correct / total, 3),
        "model_pct": round(model_correct / total, 3),
        "overrides_total": om,
        "saved": saved,
        "hurt": hurt,
    }
