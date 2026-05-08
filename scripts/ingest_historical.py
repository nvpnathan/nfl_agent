#!/usr/bin/env python3
"""One-time script to ingest 2018-2025 historical game data into SQLite."""
import pandas as pd
import yaml
from src.db.schema import create_schema
from src.db.queries import insert_game
from src.data.historical import load_games

def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    db_path = config["db"]["path"]
    create_schema(db_path)

    seasons = list(range(2018, 2026))
    print(f"Loading seasons: {seasons}")
    games_df = load_games(seasons)
    print(f"Loaded {len(games_df)} completed games")

    for _, row in games_df.iterrows():
        # nfl_data_py returns 'gameday'; fall back to 'game_date' if renamed
        game_date_col = "game_date" if "game_date" in row.index else "gameday"
        insert_game(db_path, {
            "game_id": row["game_id"],
            "season": int(row["season"]),
            "week": int(row["week"]),
            "game_type": row.get("game_type", "regular"),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "game_date": str(row[game_date_col]),
            "stadium": row.get("stadium"),
            "is_outdoor": int(not str(row.get("roof", "outdoors")).startswith("dome")),
            "home_score": int(row["home_score"]) if pd.notna(row["home_score"]) else None,
            "away_score": int(row["away_score"]) if pd.notna(row["away_score"]) else None,
            "home_win": int(row["home_win"]),
        })
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
