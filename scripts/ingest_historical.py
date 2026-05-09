#!/usr/bin/env python3
"""
Backfill ESPN game data for 2018-2025 into SQLite.
Checkpoints on espn_id: already-ingested games are skipped.
Rate: 1 req/sec. Estimated time: ~50 minutes for full backfill.
"""
import time
import yaml
from src.db.schema import create_schema
from src.db.queries import (
    get_existing_espn_ids, insert_espn_game, insert_team_stats, insert_injury,
)
from src.data.espn import (
    fetch_scoreboard, fetch_game_summary, parse_game,
    parse_box_score, parse_game_injuries,
)

SEASONS = list(range(2018, 2026))
# (season_type, week_range, label)
WEEK_CONFIGS = [
    (2, range(1, 23), "regular"),   # regular season; ESPN returns empty for non-existent weeks
    (3, range(1, 6), "postseason"),
]


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    db_path = config["db"]["path"]

    create_schema(db_path)
    existing = get_existing_espn_ids(db_path)
    print(f"Checkpoint: {len(existing)} games already in DB.")

    total_new = 0
    for season in SEASONS:
        for season_type, weeks, label in WEEK_CONFIGS:
            for week in weeks:
                try:
                    events = fetch_scoreboard(season, week, season_type)
                except Exception as e:
                    print(f"  Scoreboard error {season} {label} wk{week}: {e}")
                    time.sleep(2)
                    continue

                if not events:
                    continue

                for event in events:
                    espn_id = str(event["id"])
                    if espn_id in existing:
                        continue

                    game = parse_game(event)
                    if game["home_win"] is None:
                        # Skip incomplete/upcoming games during historical backfill
                        continue

                    insert_espn_game(db_path, game)

                    try:
                        time.sleep(1.0)
                        summary = fetch_game_summary(espn_id)
                        for stats in parse_box_score(summary, espn_id):
                            if stats["team"]:
                                insert_team_stats(db_path, stats)
                        for injury in parse_game_injuries(
                            summary, game["season"], game["week"]
                        ):
                            if injury["athlete_id"]:
                                insert_injury(db_path, injury)
                    except Exception as e:
                        print(f"  Warning: summary failed for {espn_id}: {e}")

                    existing.add(espn_id)
                    total_new += 1

                print(
                    f"  {season} {label} week {week}: "
                    f"+{len([e for e in events if str(e['id']) in existing])} games"
                )

    print(f"\nIngestion complete. Added {total_new} new games.")


if __name__ == "__main__":
    main()
