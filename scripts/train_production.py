#!/usr/bin/env python3
import argparse
import json
import sqlite3
import yaml

from src.db.queries import insert_model_training_run
from src.features.builder import build_training_dataset
from src.model.train import train_model


def _seasons_with_completed_games_and_odds(db_path: str) -> list[int]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT g.season
            FROM games g
            JOIN game_odds o ON o.espn_id = g.espn_id
            WHERE g.home_win IS NOT NULL
            ORDER BY g.season
            """
        ).fetchall()
        return [int(r[0]) for r in rows]
    finally:
        conn.close()


def _parse_seasons(raw: str) -> list[int]:
    values = [s.strip() for s in raw.split(",") if s.strip()]
    return [int(s) for s in values]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one production model artifact (no backtest folds)."
    )
    parser.add_argument(
        "--from-config",
        action="store_true",
        help="Use model.train_seasons from config.yaml instead of auto-discovered seasons.",
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Comma-separated seasons override, e.g. 2018,2019,2020,2021,2022,2023,2024,2025",
    )
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    db_path = config["db"]["path"]
    model_path = config["model"]["path"]

    if args.seasons:
        seasons = _parse_seasons(args.seasons)
        source = "--seasons"
    elif args.from_config:
        seasons = [int(s) for s in config["model"]["train_seasons"]]
        source = "config.model.train_seasons"
    else:
        seasons = _seasons_with_completed_games_and_odds(db_path)
        source = "auto-discovered completed seasons with odds"

    if not seasons:
        raise SystemExit("No eligible seasons found. Ensure DB has completed games with odds.")

    print(f"Training production artifact → {model_path}")
    print(f"Season source: {source}")
    print(f"Seasons: {seasons}")

    df = build_training_dataset(db_path=db_path, seasons=seasons)
    if df.empty:
        raise SystemExit("Training dataset is empty. Nothing to train.")

    print(f"Rows: {len(df)}")
    by_season = df.groupby("season").size().to_dict()
    print(f"Rows by season: {by_season}")

    metrics = train_model(df, model_path)
    run_id = insert_model_training_run(db_path, {
        "model_version": metrics["model_version"],
        "cv_accuracy_mean": metrics["cv_accuracy_mean"],
        "cv_accuracy_std": metrics["cv_accuracy_std"],
        "n_samples": metrics["n_samples"],
        "seasons_used": json.dumps(seasons),
    })
    print(f"Training run recorded: id={run_id}")
    print("Training complete:")
    print(metrics)


if __name__ == "__main__":
    main()
