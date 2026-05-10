import sqlite3


def create_schema(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS games (
            espn_id TEXT PRIMARY KEY,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            game_type TEXT NOT NULL DEFAULT 'regular',
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_espn_id TEXT,
            away_espn_id TEXT,
            game_date TEXT NOT NULL,
            venue TEXT,
            is_indoor INTEGER DEFAULT 0,
            is_neutral INTEGER DEFAULT 0,
            attendance INTEGER,
            home_score INTEGER,
            away_score INTEGER,
            home_win INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS team_game_stats (
            espn_id TEXT NOT NULL,
            team TEXT NOT NULL,
            is_home INTEGER NOT NULL,
            total_yards INTEGER,
            pass_yards INTEGER,
            rush_yards INTEGER,
            turnovers INTEGER,
            first_downs INTEGER,
            third_down_att INTEGER,
            third_down_made INTEGER,
            red_zone_att INTEGER,
            red_zone_made INTEGER,
            possession_secs INTEGER,
            sacks_taken INTEGER,
            PRIMARY KEY (espn_id, team),
            FOREIGN KEY (espn_id) REFERENCES games(espn_id)
        );

        CREATE TABLE IF NOT EXISTS injury_reports (
            injury_id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            team TEXT NOT NULL,
            athlete_id TEXT NOT NULL,
            athlete_name TEXT NOT NULL,
            position TEXT,
            status TEXT,
            is_qb INTEGER DEFAULT 0,
            fetched_at TEXT DEFAULT (datetime('now')),
            UNIQUE(season, week, athlete_id)
        );

        CREATE TABLE IF NOT EXISTS depth_charts (
            depth_chart_id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            team TEXT NOT NULL,
            athlete_id TEXT NOT NULL,
            athlete_name TEXT NOT NULL,
            rank INTEGER NOT NULL,
            fetched_at TEXT DEFAULT (datetime('now')),
            UNIQUE(season, week, team, rank)
        );

        CREATE TABLE IF NOT EXISTS game_odds (
            espn_id TEXT PRIMARY KEY,
            home_spread REAL NOT NULL,
            game_total REAL NOT NULL,
            home_moneyline TEXT,
            away_moneyline TEXT,
            fetched_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (espn_id) REFERENCES games(espn_id)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            home_win_prob REAL NOT NULL,
            odds_implied_prob REAL,
            model_version TEXT,
            predicted_winner TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (game_id) REFERENCES games(espn_id)
        );

        CREATE TABLE IF NOT EXISTS weekly_assignments (
            assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            game_id TEXT NOT NULL,
            predicted_winner TEXT NOT NULL,
            confidence_points INTEGER NOT NULL,
            win_probability REAL NOT NULL,
            is_uncertain INTEGER DEFAULT 0,
            is_overridden INTEGER DEFAULT 0,
            override_reason TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(season, week, game_id)
        );

        CREATE TABLE IF NOT EXISTS weekly_submissions (
            submission_id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'locked',
            source TEXT NOT NULL DEFAULT 'streamlit',
            submitted_at TEXT DEFAULT (datetime('now')),
            UNIQUE(season, week)
        );

        CREATE TABLE IF NOT EXISTS weekly_submission_picks (
            submission_id INTEGER NOT NULL,
            game_id TEXT NOT NULL,
            model_pick TEXT NOT NULL,
            submitted_pick TEXT NOT NULL,
            model_points INTEGER NOT NULL,
            submitted_points INTEGER NOT NULL,
            points_delta INTEGER NOT NULL,
            win_probability REAL NOT NULL,
            tier TEXT NOT NULL,
            market TEXT,
            is_overridden INTEGER DEFAULT 0,
            override_reason TEXT,
            PRIMARY KEY (submission_id, game_id),
            FOREIGN KEY (submission_id) REFERENCES weekly_submissions(submission_id)
                ON DELETE CASCADE,
            FOREIGN KEY (game_id) REFERENCES games(espn_id)
        );

        CREATE TABLE IF NOT EXISTS conversations (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            season INTEGER,
            week INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS rerankings (
            reranking_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            game_id TEXT NOT NULL,
            old_points INTEGER NOT NULL,
            new_points INTEGER NOT NULL,
            reason TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS model_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            model_version TEXT,
            accuracy REAL,
            brier_score REAL,
            expected_points REAL,
            actual_points REAL,
            baseline_accuracy REAL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS model_training_runs (
            model_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_version TEXT NOT NULL,
            cv_accuracy_mean REAL NOT NULL,
            cv_accuracy_std REAL NOT NULL,
            n_samples INTEGER NOT NULL,
            seasons_used TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        DROP TABLE IF EXISTS family_picks;
        """)
        conn.commit()
    finally:
        conn.close()
