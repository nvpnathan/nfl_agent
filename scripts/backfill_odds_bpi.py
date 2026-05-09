#!/usr/bin/env python3
"""Backfill 2024 game odds using ESPN BPI win probability as a spread proxy.

Used for games where Bet365 historical data is unavailable in ESPN's API
(most 2024 regular season and playoff games). The BPI-implied spread is
within ~1 point of market spreads on validated 2024 games.

Conversion: implied_spread = -(bpi_win_prob - 0.5) / 0.028
game_total defaults to 45.0 (2024 season average from available Bet365 data).

Safe to re-run — skips games already in game_odds.
"""
import sqlite3
import time
import yaml
import httpx
from src.db.queries import insert_game_odds, get_game_odds

GAME_TOTAL_DEFAULT = 45.0
BPI_SCALE = 0.028  # each point of spread ≈ 2.8% win probability


def fetch_bpi_win_prob(espn_id: str) -> float | None:
    url = (
        f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
        f"/events/{espn_id}/competitions/{espn_id}/predictor"
    )
    with httpx.Client(timeout=10) as client:
        resp = client.get(url)
    if resp.status_code != 200:
        return None
    data = resp.json()
    stats = {
        s["name"]: s["value"]
        for s in data.get("homeTeam", {}).get("statistics", [])
    }
    proj = stats.get("gameProjection")
    return proj / 100.0 if proj is not None else None


def bpi_to_spread(win_prob: float) -> float:
    return -round((win_prob - 0.5) / BPI_SCALE, 1)


def main() -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    db_path = config["db"]["path"]

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("""
            SELECT g.espn_id FROM games g
            LEFT JOIN game_odds o USING(espn_id)
            WHERE g.season=2024 AND g.home_win IS NOT NULL AND o.espn_id IS NULL
            ORDER BY g.game_date
        """).fetchall()
    espn_ids = [r[0] for r in rows]
    print(f"Found {len(espn_ids)} 2024 games needing BPI odds")

    fetched, missing = 0, 0
    for espn_id in espn_ids:
        if get_game_odds(db_path, espn_id):
            continue
        win_prob = fetch_bpi_win_prob(espn_id)
        if win_prob is None:
            missing += 1
            time.sleep(0.3)
            continue
        insert_game_odds(db_path, {
            "espn_id": espn_id,
            "home_spread": bpi_to_spread(win_prob),
            "game_total": GAME_TOTAL_DEFAULT,
            "home_moneyline": None,
            "away_moneyline": None,
        })
        fetched += 1
        time.sleep(0.3)

    print(f"Done. fetched={fetched} missing={missing}")


if __name__ == "__main__":
    main()
