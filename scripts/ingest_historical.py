#!/usr/bin/env python3
"""
Ingest ESPN game data into SQLite.

Modes:
  full    — Backfill all seasons (2018-2025), ~50 min
  week    — Ingest a specific season/week (default: current)
  weeks   — Ingest the last N completed weeks

Already-ingested games are skipped. Box scores and injuries are always
refreshed for existing games (INSERT OR REPLACE). Rate-limited to 1 req/sec.

Usage:
    uv run python scripts/ingest_historical.py                    # full backfill
    uv run python scripts/ingest_historical.py --week 2026 14     # specific week
    uv run python scripts/ingest_historical.py --weeks 2          # last 2 weeks
"""
import argparse
import time
from datetime import datetime, timezone

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
WEEK_CONFIGS = [
    (2, range(1, 23), "regular"),
    (3, range(1, 6), "postseason"),
]

# Days since epoch for season boundary heuristic (mid-August)
_SEASON_BOUNDARY = 1756089600  # 2025-08-25 UTC


def _detect_current_week(season: int) -> tuple[int, int]:
    """Try to detect the current live week via ESPN scoreboard."""
    for season_type in (2, 3):
        for week in range(1, 19 if season_type == 2 else 6):
            try:
                events = fetch_scoreboard(season, week, season_type)
                if not events:
                    continue
                statuses = [
                    e.get("status", {}).get("type", {}).get("name", "")
                    for e in events
                ]
                has_upcoming = any(
                    "SCHEDULED" in s or "IN" in s for s in statuses
                )
                has_final = any("FINAL" in s for s in statuses) or True
                if has_upcoming or (season_type == 3):
                    return week, season_type
            except Exception:
                continue
    return 1, 2


def _ingest_week(db_path: str, season: int, week: int,
                 season_type: int) -> tuple[int, int]:
    """Ingest a single week. Returns (new_games, total_refreshed)."""
    existing = set()
    try:
        with open(db_path) as f:
            pass
    except FileNotFoundError:
        existing = set()

    total_new = 0
    total_refreshed = 0

    try:
        events = fetch_scoreboard(season, week, season_type)
    except Exception as e:
        print(f"  Scoreboard error {season} wk{week}: {e}")
        time.sleep(2)
        return 0, 0

    if not events:
        return 0, 0

    # Load existing IDs once for this week's scope
    all_existing = get_existing_espn_ids(db_path)

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    for event in events:
        espn_id = str(event["id"])
        game_already_exists = espn_id in all_existing

        game = parse_game(event)
        if not game_already_exists and game["home_win"] is None:
            continue

        if not game_already_exists:
            insert_espn_game(db_path, game)
            total_new += 1

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

        all_existing.add(espn_id)
        total_refreshed += 1

    return total_new, total_refreshed


def _run_full(db_path: str) -> None:
    """Full historical backfill."""
    create_schema(db_path)
    existing = get_existing_espn_ids(db_path)
    print(f"Checkpoint: {len(existing)} games already in DB.")

    total_new = 0
    for season in SEASONS:
        for season_type, weeks, label in WEEK_CONFIGS:
            for week in weeks:
                new_count, _ = _ingest_week(db_path, season, week, season_type)
                total_new += new_count

    print(f"\nIngestion complete. Added {total_new} new games.")


def _run_week(db_path: str, season: int, week: int) -> None:
    """Ingest a specific season/week."""
    create_schema(db_path)
    existing = get_existing_espn_ids(db_path)
    print(f"Checkpoint: {len(existing)} games already in DB.")

    total_new = 0
    total_refreshed = 0

    for season_type in (2, 3):
        new_count, refreshed = _ingest_week(db_path, season, week, season_type)
        total_new += new_count
        total_refreshed += refreshed

    print(f"\nIngested {total_new} new games, refreshed {total_refreshed}.")


def _run_weeks(db_path: str, n: int) -> None:
    """Ingest the last N completed weeks."""
    create_schema(db_path)
    existing = get_existing_espn_ids(db_path)
    print(f"Checkpoint: {len(existing)} games already in DB.")

    now = datetime.now(timezone.utc)
    season = now.year if now.month >= 8 else now.year - 1

    # Try to detect the current week, then go backwards
    detected_week, season_type = _detect_current_week(season)

    total_new = 0
    total_refreshed = 0

    if detected_week > 0:
        for offset in range(n):
            week = max(1, detected_week - offset)
            new_count, refreshed = _ingest_week(db_path, season, week, season_type)
            total_new += new_count
            total_refreshed += refreshed
            print(f"  Week {week} ({season_type}): +{new_count} new, {refreshed} refreshed")
    else:
        print("  Could not detect current week. Try --week <season> <week>.")

    print(f"\nIngested {total_new} new games, refreshed {total_refreshed}.")


def main() -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    db_path = config["db"]["path"]

    parser = argparse.ArgumentParser(
        description="Ingest ESPN game data. Use --help for modes."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--full", action="store_true",
        help="Full backfill all seasons 2018-2025 (~50 min)",
    )
    group.add_argument(
        "--week", nargs=2, type=int, metavar=("SEASON", "WEEK"),
        help="Ingest a specific season/week, e.g. 2026 14",
    )
    group.add_argument(
        "--weeks", type=int, default=None, metavar="N",
        help="Ingest the last N completed weeks (auto-detects current week)",
    )

    args = parser.parse_args()

    if args.full:
        _run_full(db_path)
    elif args.week is not None:
        season, week = args.week
        _run_week(db_path, season, week)
    elif args.weeks is not None:
        _run_weeks(db_path, args.weeks)


if __name__ == "__main__":
    main()
