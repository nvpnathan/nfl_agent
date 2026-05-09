#!/usr/bin/env python3
"""Backfill Bet365 pre-match odds from ESPN for all games with season >= 2019.

Run after ingest_historical.py. Safe to re-run — skips games already in game_odds.
Rate-limited to 2 req/sec. Expect ~20 minutes for a full backfill of 2019-2024.
"""
import sqlite3
import time
import yaml
from src.db.schema import create_schema
from src.db.queries import insert_game_odds, get_game_odds
from src.data.espn import fetch_game_odds


def main() -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    db_path = config["db"]["path"]
    create_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT espn_id FROM games WHERE season >= 2019 AND home_win IS NOT NULL ORDER BY season, week"
        ).fetchall()
    espn_ids = [r[0] for r in rows]
    print(f"Found {len(espn_ids)} completed games from 2019+")

    already, fetched, missing = 0, 0, 0
    for espn_id in espn_ids:
        if get_game_odds(db_path, espn_id):
            already += 1
            continue
        odds = fetch_game_odds(espn_id)
        if odds:
            insert_game_odds(db_path, odds)
            fetched += 1
        else:
            missing += 1
        time.sleep(0.5)

    print(f"Done. already={already} fetched={fetched} no_data={missing}")


if __name__ == "__main__":
    main()
