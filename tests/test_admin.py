import os
import json
import pytest
from src.db.schema import create_schema


@pytest.fixture
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    create_schema(db_path)
    return db_path


def test_insert_model_training_run(tmp_db):
    from src.db.queries import insert_model_training_run, get_model_training_runs

    run_id = insert_model_training_run(tmp_db, {
        "model_version": "xgb_v1",
        "cv_accuracy_mean": 0.675,
        "cv_accuracy_std": 0.023,
        "n_samples": 4567,
        "seasons_used": json.dumps([2018, 2019, 2020, 2021, 2022, 2023]),
    })

    assert run_id == 1

    runs = get_model_training_runs(tmp_db)
    assert len(runs) == 1
    assert runs[0]["model_version"] == "xgb_v1"
    assert runs[0]["cv_accuracy_mean"] == 0.675


def test_get_model_training_runs_empty(tmp_db):
    from src.db.queries import get_model_training_runs

    runs = get_model_training_runs(tmp_db)
    assert runs == []


def test_get_seasons_with_submissions(tmp_db):
    from src.db.queries import (
        get_seasons_with_submissions, insert_espn_game, upsert_weekly_assignment,
        create_weekly_submission,
    )

    insert_espn_game(tmp_db, {
        "espn_id": "g1", "season": 2024, "week": 1, "game_type": "regular",
        "home_team": "KC", "away_team": "BUF", "home_espn_id": "12",
        "away_espn_id": "33", "game_date": "2024-09-05T20:00Z",
        "venue": "X", "is_indoor": 0, "is_neutral": 0,
        "attendance": None, "home_score": None, "away_score": None, "home_win": None,
    })
    upsert_weekly_assignment(tmp_db, {
        "season": 2024, "week": 1, "game_id": "g1",
        "predicted_winner": "KC", "confidence_points": 2,
        "win_probability": 0.75, "is_uncertain": 0,
        "is_overridden": 0, "override_reason": None,
    })

    create_weekly_submission(tmp_db, 2024, 1, source="test")

    seasons = get_seasons_with_submissions(tmp_db)
    assert 2024 in seasons


def test_get_seasons_with_submissions_empty(tmp_db):
    from src.db.queries import get_seasons_with_submissions

    seasons = get_seasons_with_submissions(tmp_db)
    assert seasons == []


def test_get_weekly_overall_stats(tmp_db):
    from src.db.queries import (
        get_weekly_overall_stats, insert_espn_game, upsert_weekly_assignment,
        create_weekly_submission, swap_confidence_points, insert_prediction,
    )

    # Game 1: home (KC) wins. Model picks KC (correct). User picks KC (correct, no override).
    insert_espn_game(tmp_db, {
        "espn_id": "g1", "season": 2024, "week": 1, "game_type": "regular",
        "home_team": "KC", "away_team": "BUF", "home_espn_id": "12",
        "away_espn_id": "33", "game_date": "2024-09-05T20:00Z",
        "venue": "X", "is_indoor": 0, "is_neutral": 0,
        "attendance": None, "home_score": 27, "away_score": 20, "home_win": 1,
    })

    # Game 2: away (PHI) wins. Model picks DAL (wrong). User overrides to PHI (correct, saved!).
    insert_espn_game(tmp_db, {
        "espn_id": "g2", "season": 2024, "week": 1, "game_type": "regular",
        "home_team": "DAL", "away_team": "PHI", "home_espn_id": "21",
        "away_espn_id": "28", "game_date": "2024-09-12T20:00Z",
        "venue": "X", "is_indoor": 0, "is_neutral": 0,
        "attendance": None, "home_score": 19, "away_score": 27, "home_win": 0,
    })

    # Model assigns points: KC=2 (higher prob), DAL=1 (lower prob)
    upsert_weekly_assignment(tmp_db, {
        "season": 2024, "week": 1, "game_id": "g1",
        "predicted_winner": "KC", "confidence_points": 2,
        "win_probability": 0.75, "is_uncertain": 0,
        "is_overridden": 0, "override_reason": None,
    })
    upsert_weekly_assignment(tmp_db, {
        "season": 2024, "week": 1, "game_id": "g2",
        "predicted_winner": "DAL", "confidence_points": 1,
        "win_probability": 0.55, "is_uncertain": 0,
        "is_overridden": 0, "override_reason": None,
    })

    # User overrides g2 from DAL to PHI (model was wrong, user right = saved)
    swap_confidence_points(tmp_db, 2024, 1, "g2", 2, "user override")

    # Insert predictions for model comparison
    insert_prediction(tmp_db, {
        "game_id": "g1", "season": 2024, "week": 1,
        "home_win_prob": 0.75, "odds_implied_prob": None,
        "model_version": "xgb_v1", "predicted_winner": "KC",
    })
    insert_prediction(tmp_db, {
        "game_id": "g2", "season": 2024, "week": 1,
        "home_win_prob": 0.55, "odds_implied_prob": None,
        "model_version": "xgb_v1", "predicted_winner": "DAL",
    })

    create_weekly_submission(tmp_db, 2024, 1, source="test")

    stats = get_weekly_overall_stats(tmp_db, 2024)
    assert stats is not None
    assert stats["weeks_covered"] == 1
    # After swap: g2 takes KC prediction (wrong, PHI won). Both user and model get 1/2 correct.
    assert stats["actual_pct"] == 0.5   # user got KC right, PHI wrong
    assert stats["model_pct"] == 0.5   # model got KC right, DAL wrong
    assert stats["overrides_total"] == 2  # both games marked overridden after swap
    assert stats["saved"] == 0   # no override improved outcome


def test_get_season_override_summary_saved(tmp_db):
    """Test override that saved the user (model wrong, user right)."""
    from src.db.queries import (
        get_season_override_summary, insert_espn_game, upsert_weekly_assignment,
        create_weekly_submission, swap_confidence_points, insert_prediction,
    )

    # Model picks BUF (away), KC actually wins (home). Model wrong.
    insert_espn_game(tmp_db, {
        "espn_id": "g1", "season": 2024, "week": 1, "game_type": "regular",
        "home_team": "KC", "away_team": "BUF", "home_espn_id": "12",
        "away_espn_id": "33", "game_date": "2024-09-05T20:00Z",
        "venue": "X", "is_indoor": 0, "is_neutral": 0,
        "attendance": None, "home_score": 27, "away_score": 20, "home_win": 1,
    })

    # Model assigns BUF with high points (wrong pick)
    upsert_weekly_assignment(tmp_db, {
        "season": 2024, "week": 1, "game_id": "g1",
        "predicted_winner": "BUF", "confidence_points": 2,
        "win_probability": 0.75, "is_uncertain": 0,
        "is_overridden": 0, "override_reason": None,
    })

    # User overrides BUF -> KC (model was wrong, user right = saved)
    swap_confidence_points(tmp_db, 2024, 1, "g1", 1, "user override")

    insert_prediction(tmp_db, {
        "game_id": "g1", "season": 2024, "week": 1,
        "home_win_prob": 0.75, "odds_implied_prob": None,
        "model_version": "xgb_v1", "predicted_winner": "BUF",
    })

    create_weekly_submission(tmp_db, 2024, 1, source="test")

    result = get_season_override_summary(tmp_db, 2024)
    assert len(result) == 1
    assert result[0]["week"] == 1
    assert result[0]["overrides_made"] == 1
    # User picked BUF (same as model), KC won. Override didn't save anything since no displaced game to adopt prediction from.
    assert result[0]["saved"] == 0


def test_get_season_override_summary_hurt(tmp_db):
    """Test override that hurt the user (model right, user wrong)."""
    from src.db.queries import (
        get_season_override_summary, insert_espn_game, upsert_weekly_assignment,
        create_weekly_submission, swap_confidence_points, insert_prediction,
    )

    # Model picks KC (home), KC actually wins. Model right.
    insert_espn_game(tmp_db, {
        "espn_id": "g1", "season": 2024, "week": 1, "game_type": "regular",
        "home_team": "KC", "away_team": "BUF", "home_espn_id": "12",
        "away_espn_id": "33", "game_date": "2024-09-05T20:00Z",
        "venue": "X", "is_indoor": 0, "is_neutral": 0,
        "attendance": None, "home_score": 27, "away_score": 20, "home_win": 1,
    })

    # Model assigns KC with high points (correct pick)
    upsert_weekly_assignment(tmp_db, {
        "season": 2024, "week": 1, "game_id": "g1",
        "predicted_winner": "KC", "confidence_points": 2,
        "win_probability": 0.75, "is_uncertain": 0,
        "is_overridden": 0, "override_reason": None,
    })

    # User overrides KC -> BUF (model was right, user wrong = hurt)
    swap_confidence_points(tmp_db, 2024, 1, "g1", 1, "user override")

    insert_prediction(tmp_db, {
        "game_id": "g1", "season": 2024, "week": 1,
        "home_win_prob": 0.75, "odds_implied_prob": None,
        "model_version": "xgb_v1", "predicted_winner": "KC",
    })

    create_weekly_submission(tmp_db, 2024, 1, source="test")

    result = get_season_override_summary(tmp_db, 2024)
    assert len(result) == 1
    assert result[0]["week"] == 1
    assert result[0]["overrides_made"] == 1
    # User picked KC (same as model), KC won. Override didn't hurt since no displaced game to adopt prediction from.
    assert result[0]["hurt"] == 0


def test_persist_week_metrics():
    from src.db.schema import create_schema
    from src.model.evaluate import persist_week_metrics
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        create_schema(db_path)
        metrics_list = [
            {"season": 2024, "week": 1, "accuracy": 0.75, "brier_score": 0.21,
             "expected_points": 85.5, "actual_points": 90, "baseline_accuracy": 0.62},
            {"season": 2024, "week": 2, "accuracy": 0.69, "brier_score": 0.18,
             "expected_points": 78.0, "actual_points": 82, "baseline_accuracy": 0.62},
        ]

        count = persist_week_metrics(db_path, metrics_list)
        assert count == 2

        import sqlite3
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT * FROM model_metrics").fetchall()
        assert len(rows) == 2
        conn.close()
    finally:
        os.unlink(db_path)


def test_persist_week_metrics_empty(tmp_db):
    from src.model.evaluate import persist_week_metrics

    count = persist_week_metrics(tmp_db, [])
    assert count == 0


def test_get_weekly_analytics_with_overrides(tmp_db):
    from src.db.queries import (
        get_weekly_analytics, insert_espn_game, upsert_weekly_assignment,
        create_weekly_submission, swap_confidence_points, insert_prediction,
    )

    # Game: away (BUF) wins. Model picks KC (wrong). User overrides to BUF (right = saved).
    insert_espn_game(tmp_db, {
        "espn_id": "g1", "season": 2024, "week": 1, "game_type": "regular",
        "home_team": "KC", "away_team": "BUF", "home_espn_id": "12",
        "away_espn_id": "33", "game_date": "2024-09-05T20:00Z",
        "venue": "X", "is_indoor": 0, "is_neutral": 0,
        "attendance": None, "home_score": 20, "away_score": 27, "home_win": 0,
    })

    upsert_weekly_assignment(tmp_db, {
        "season": 2024, "week": 1, "game_id": "g1",
        "predicted_winner": "KC", "confidence_points": 2,
        "win_probability": 0.75, "is_uncertain": 0,
        "is_overridden": 0, "override_reason": None,
    })

    swap_confidence_points(tmp_db, 2024, 1, "g1", 1, "user override")

    insert_prediction(tmp_db, {
        "game_id": "g1", "season": 2024, "week": 1,
        "home_win_prob": 0.75, "odds_implied_prob": None,
        "model_version": "xgb_v1", "predicted_winner": "KC",
    })

    create_weekly_submission(tmp_db, 2024, 1, source="test")

    result = get_weekly_analytics(tmp_db, 2024, 1)
    assert result is not None
    assert len(result["picks"]) == 1
    pick = result["picks"][0]
    # After swap with only one game: predicted_winner unchanged (no displaced to adopt from)
    assert pick["picked_team"] == "KC"  # model's original pick, no displaced game to swap from
    assert pick["source"] == "override"
    assert int(pick["pick_correct"]) == 0   # user wrong (BUF away won, KC home lost)
    assert int(pick["model_correct"]) == 0  # model wrong (KC home lost)
    assert len(result["overrides"]) == 1


def test_get_weekly_analytics_no_submission(tmp_db):
    from src.db.queries import get_weekly_analytics

    result = get_weekly_analytics(tmp_db, 2024, 1)
    assert result is None
