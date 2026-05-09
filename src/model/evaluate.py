from sklearn.metrics import brier_score_loss


def baseline_accuracy(predictions: list[dict]) -> float:
    if not predictions:
        return 0.0
    return sum(1 for p in predictions if bool(p["home_win"])) / len(predictions)


def compute_week_metrics(predictions: list[dict]) -> dict:
    n = len(predictions)
    if n == 0:
        return {}
    correct = sum(
        1 for p in predictions
        if (p["home_win_prob"] >= 0.5) == bool(p["home_win"])
    )
    actual_pts = sum(
        p["confidence_points"] for p in predictions
        if (p["home_win_prob"] >= 0.5) == bool(p["home_win"])
    )
    probs = [p["home_win_prob"] for p in predictions]
    labels = [p["home_win"] for p in predictions]
    return {
        "accuracy": correct / n,
        "actual_points": actual_pts,
        "expected_points": sum(
            p.get("win_probability", p["home_win_prob"]) * p["confidence_points"]
            for p in predictions
        ),
        "brier_score": brier_score_loss(labels, probs),
        "baseline_accuracy": sum(1 for p in predictions if bool(p["home_win"])) / n,
    }


def run_season_backtest(
    model_path: str,
    seasons: list[int],
    db_path: str,
    point_range: tuple = (1, 16),
) -> list[dict]:
    from src.features.builder import build_training_dataset, FEATURE_COLS
    from src.model.predict import predict_week
    from src.optimizer.confidence import assign_confidence_points

    all_metrics = []
    for season in seasons:
        games_df = build_training_dataset(db_path=db_path, seasons=[season])
        weeks = sorted(games_df["week"].unique())
        for week in weeks:
            week_games = games_df[games_df["week"] == week]
            game_inputs = []
            for _, row in week_games.iterrows():
                game_inputs.append({
                    "espn_id": row["espn_id"],
                    "home_team": row.get("home_team", ""),
                    "away_team": row.get("away_team", ""),
                    "features": {col: row[col] for col in FEATURE_COLS if col in row.index},
                })
            if not game_inputs:
                continue
            predictions = predict_week(model_path, game_inputs)
            assignments = assign_confidence_points(predictions, point_range)
            assign_by_id = {a["espn_id"]: a["confidence_points"] for a in assignments}
            for pred in predictions:
                pred["confidence_points"] = assign_by_id[pred["espn_id"]]
                eid = pred["espn_id"]
                home_win_rows = week_games[week_games["espn_id"] == eid]["home_win"]
                if len(home_win_rows) == 0:
                    raise ValueError(f"espn_id {eid} not found in week_games")
                pred["home_win"] = int(home_win_rows.values[0])
            metrics = compute_week_metrics(predictions)
            metrics["season"] = season
            metrics["week"] = week
            all_metrics.append(metrics)
    return all_metrics
