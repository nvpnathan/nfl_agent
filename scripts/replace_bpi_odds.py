#!/usr/bin/env python3
"""Replace BPI-derived odds with real Odds API historical spreads and totals.

Targets all game_odds entries where home_moneyline IS NULL (BPI-derived).
Queries the Odds API historical endpoint once per (season, week) — 36 total calls.
Uses DraftKings as primary bookmaker, falls back to first available.
Safe to re-run: only overwrites BPI entries (home_moneyline IS NULL).
"""
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import requests
import yaml

TEAM_NAME_MAP = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WSH",
}


def _parse_week_odds(data: list[dict]) -> dict[tuple, dict]:
    """Index Odds API response by (home_abbr, away_abbr) -> {home_spread, game_total}."""
    result = {}
    for game in data:
        home_abbr = TEAM_NAME_MAP.get(game["home_team"])
        away_abbr = TEAM_NAME_MAP.get(game["away_team"])
        if not home_abbr or not away_abbr:
            continue
        bk = next(
            (b for b in game.get("bookmakers", []) if b["key"] == "draftkings"),
            next(iter(game.get("bookmakers", [])), None),
        )
        if not bk:
            continue
        markets = {m["key"]: m for m in bk.get("markets", [])}
        spread_market = markets.get("spreads")
        if not spread_market:
            continue
        home_outcome = next(
            (o for o in spread_market["outcomes"] if TEAM_NAME_MAP.get(o["name"]) == home_abbr),
            None,
        )
        if home_outcome is None:
            continue
        total_market = markets.get("totals")
        game_total = total_market["outcomes"][0]["point"] if total_market else 45.0
        result[(home_abbr, away_abbr)] = {
            "home_spread": float(home_outcome["point"]),
            "game_total": float(game_total),
        }
    return result


def fetch_week_odds(api_key: str, query_date: str) -> dict[tuple, dict]:
    resp = requests.get(
        "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/odds/",
        params={"apiKey": api_key, "regions": "us", "markets": "spreads,totals", "date": query_date},
        timeout=15,
    )
    if not resp.ok:
        print(f"  API error {resp.status_code}: {resp.text[:200]}")
        return {}
    remaining = resp.headers.get("x-requests-remaining", "?")
    data = resp.json().get("data", [])
    print(f"  {len(data)} games from API  (remaining credits: {remaining})")
    return _parse_week_odds(data)


def main() -> None:
    api_key = open(".env").read().strip().split("=")[1]
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    db_path = config["db"]["path"]

    # Get all BPI-derived entries grouped by season+week
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        week_groups = conn.execute("""
            SELECT g.season, g.week, MIN(g.game_date) as first_game
            FROM games g JOIN game_odds o USING(espn_id)
            WHERE g.season >= 2024 AND o.home_moneyline IS NULL
            GROUP BY g.season, g.week
            ORDER BY g.season, g.week
        """).fetchall()

    print(f"Replacing BPI odds across {len(week_groups)} week groups...\n")
    replaced, unmatched = 0, 0

    for group in week_groups:
        season, week, first_game = group["season"], group["week"], group["first_game"]
        # Query 2 days before first kickoff of the week
        dt = datetime.fromisoformat(first_game.replace("Z", "+00:00"))
        query_date = (dt - timedelta(days=2)).strftime("%Y-%m-%dT12:00:00Z")
        print(f"Season {season} week {week}  (query: {query_date})")

        odds_map = fetch_week_odds(api_key, query_date)

        # Fetch all BPI games for this week
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            games = conn.execute("""
                SELECT g.espn_id, g.home_team, g.away_team
                FROM games g JOIN game_odds o USING(espn_id)
                WHERE g.season=? AND g.week=? AND o.home_moneyline IS NULL
            """, (season, week)).fetchall()

        with sqlite3.connect(db_path) as conn:
            for game in games:
                key = (game["home_team"], game["away_team"])
                odds = odds_map.get(key)
                if odds is None:
                    unmatched += 1
                    continue
                conn.execute("""
                    INSERT OR REPLACE INTO game_odds
                    (espn_id, home_spread, game_total, home_moneyline, away_moneyline)
                    VALUES (?, ?, ?, NULL, NULL)
                """, (game["espn_id"], odds["home_spread"], odds["game_total"]))
                replaced += 1

        time.sleep(0.5)

    print(f"\nDone. replaced={replaced}  unmatched={unmatched}")


if __name__ == "__main__":
    main()
