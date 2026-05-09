#!/usr/bin/env python3
"""Weekly refresh: fetch current week's games, odds, injuries, weather,
generate predictions, and save confidence assignments to DB.

Free data only — ESPN scoreboard, ESPN odds (BPI fallback), Open-Meteo.
Safe to re-run: upserts predictions and assignments idempotently.

Usage:
    uv run python scripts/refresh_weekly.py
    uv run python scripts/refresh_weekly.py --season 2025 --week 14
"""
import argparse
import sqlite3
import time
import yaml
from datetime import datetime, timezone

from src.data.espn import (
    fetch_scoreboard, parse_game, fetch_game_odds,
    fetch_team_injuries, parse_game_injuries,
)
from src.data.weather import get_stadium_weather
from src.db.queries import (
    insert_espn_game, insert_game_odds, insert_injury,
    get_game_odds, upsert_weekly_assignment,
)
from src.db.schema import create_schema
from src.features.builder import build_features_for_game
from src.model.predict import predict_week
from src.optimizer.confidence import assign_confidence_points, get_point_range


_POSTSEASON_SEASON_TYPES = {
    "wildcard": 3, "divisional": 3, "conference": 3, "superbowl": 3,
}
PLAYOFF_TYPES = {"wildcard", "divisional", "conference", "superbowl"}


def _current_season() -> int:
    now = datetime.now(timezone.utc)
    # NFL season year = calendar year if month >= 8, else prior year
    return now.year if now.month >= 8 else now.year - 1


def _fetch_current_week(season: int) -> tuple[int, int]:
    """Return (week, season_type) for the current NFL week by trying regular
    then postseason until we find games. Returns (0, 2) if nothing found."""
    # Try regular season weeks 1-18
    for week in range(1, 19):
        events = fetch_scoreboard(season, week, season_type=2)
        if events:
            # Find the latest week that has upcoming or recent games
            statuses = [e.get("status", {}).get("type", {}).get("name", "") for e in events]
            has_upcoming = any("SCHEDULED" in s or "IN" in s for s in statuses)
            has_final = any("FINAL" in s for s in statuses)
            if has_upcoming or (has_final and week == 1):
                # If all final, keep scanning for upcoming
                if not has_upcoming and has_final:
                    continue
                return week, 2
    # Try postseason
    for week in range(1, 6):
        events = fetch_scoreboard(season, week, season_type=3)
        if events:
            return week, 3
    return 1, 2


def fetch_odds_for_game(espn_id: str, game: dict) -> dict | None:
    """Try ESPN odds API; fall back to BPI win probability."""
    odds = fetch_game_odds(espn_id)
    if odds:
        return odds

    # BPI fallback
    import httpx
    url = (
        f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
        f"/events/{espn_id}/competitions/{espn_id}/predictor"
    )
    try:
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
        if proj is None:
            return None
        win_prob = proj / 100.0
        home_spread = -round((win_prob - 0.5) / 0.028, 1)
        return {
            "espn_id": espn_id,
            "home_spread": home_spread,
            "game_total": 45.0,
            "home_moneyline": None,
            "away_moneyline": None,
        }
    except Exception:
        return None


def fetch_injuries_for_game(db_path: str, game: dict, season: int, week: int) -> None:
    """Fetch and store current injury reports for both teams in a game."""
    for side in ("home", "away"):
        team_espn_id = game.get(f"{side}_espn_id")
        if not team_espn_id:
            continue
        try:
            raw = fetch_team_injuries(team_espn_id)
            # Parse into our schema format
            for item in raw:
                athlete = item.get("athlete", {})
                position = athlete.get("position", {}).get("abbreviation", "")
                team_abbr = game[f"{side}_team"]
                athlete_id = str(athlete.get("id", ""))
                if not athlete_id:
                    continue
                insert_injury(db_path, {
                    "season": season,
                    "week": week,
                    "team": team_abbr,
                    "athlete_id": athlete_id,
                    "athlete_name": athlete.get("displayName", ""),
                    "position": position,
                    "status": item.get("status", ""),
                    "is_qb": int(position == "QB"),
                })
        except Exception as e:
            print(f"    Warning: injury fetch failed for {game[f'{side}_team']}: {e}")
        time.sleep(0.2)


def main(season: int | None = None, week: int | None = None) -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    db_path = config["db"]["path"]
    model_path = config["model"]["path"]
    uncertainty_threshold = config.get("model", {}).get("uncertainty_threshold", 0.03)

    create_schema(db_path)

    if season is None:
        season = _current_season()

    print(f"Refreshing season {season}...")

    # Detect week if not specified
    if week is None:
        week, season_type = _fetch_current_week(season)
        print(f"  Detected: week {week}, season_type {season_type}")
    else:
        season_type = 2  # assume regular unless overridden

    # Fetch games for the week
    events = fetch_scoreboard(season, week, season_type=season_type)
    if not events:
        # Try the other season type
        alt_type = 3 if season_type == 2 else 2
        events = fetch_scoreboard(season, week, season_type=alt_type)
        if events:
            season_type = alt_type

    if not events:
        print(f"  No games found for season={season} week={week}. Exiting.")
        return

    print(f"  Found {len(events)} games for week {week}")

    # Upsert games into DB
    games_data = []
    for event in events:
        g = parse_game(event)
        insert_espn_game(db_path, g)
        games_data.append(g)

    # Fetch odds + injuries for each game
    print(f"  Fetching odds and injuries...")
    for g in games_data:
        espn_id = g["espn_id"]

        # Odds
        existing_odds = get_game_odds(db_path, espn_id)
        if not existing_odds:
            odds = fetch_odds_for_game(espn_id, g)
            if odds:
                insert_game_odds(db_path, odds)
            else:
                print(f"    No odds available for {g['home_team']} vs {g['away_team']}")
        time.sleep(0.3)

        # Injuries
        fetch_injuries_for_game(db_path, g, season, week)

    # Build features and predict
    print(f"  Building features and generating predictions...")
    games_with_features = []
    for g in games_data:
        odds = get_game_odds(db_path, g["espn_id"])
        if not odds:
            print(f"    Skipping {g['home_team']} vs {g['away_team']} — no odds")
            continue

        game_date_str = g["game_date"][:10] if g.get("game_date") else None
        weather = None
        if game_date_str and not g.get("is_indoor"):
            weather = get_stadium_weather(g["home_team"], game_date_str)

        try:
            features = build_features_for_game(
                g, db_path,
                home_spread=odds["home_spread"],
                game_total=odds["game_total"],
                weather=weather,
            )
            games_with_features.append({
                "espn_id": g["espn_id"],
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "game_date": g["game_date"],
                "game_type": g.get("game_type", "regular"),
                "features": features,
            })
        except Exception as e:
            print(f"    Feature build failed for {g['home_team']} vs {g['away_team']}: {e}")

    if not games_with_features:
        print("  No games with features — nothing to predict.")
        return

    predictions = predict_week(model_path, games_with_features)

    # Assign confidence points
    game_type = games_data[0].get("game_type", "regular") if games_data else "regular"
    is_playoff = game_type in PLAYOFF_TYPES
    n_games = len(predictions)
    point_range = get_point_range(n_games, "playoff" if is_playoff else "regular")

    pool_games = [
        {
            "game_id": p["espn_id"],
            "home_team": p["home_team"],
            "away_team": p["away_team"],
            "predicted_winner": p["predicted_winner"],
            "win_probability": p["win_probability"],
        }
        for p in predictions
    ]
    assignments = assign_confidence_points(pool_games, point_range, uncertainty_threshold)

    # Persist assignments
    for a in assignments:
        upsert_weekly_assignment(db_path, {
            "season": season,
            "week": week,
            "game_id": a["game_id"],
            "predicted_winner": a["predicted_winner"],
            "confidence_points": a["confidence_points"],
            "win_probability": a["win_probability"],
            "is_uncertain": int(a["is_uncertain"]),
            "is_overridden": 0,
            "override_reason": None,
        })

    # Print summary
    print(f"\n{'='*60}")
    print(f"WEEK {week} PICKS — Season {season}")
    print(f"{'='*60}")
    sorted_assigns = sorted(assignments, key=lambda x: x["confidence_points"], reverse=True)
    for a in sorted_assigns:
        uncertain_flag = " ⚠" if a["is_uncertain"] else ""
        print(
            f"  {a['confidence_points']:>2}pts  {a['predicted_winner']:<4}  "
            f"({a['win_probability']:.0%}){uncertain_flag}"
            f"  [{a['home_team']} vs {a['away_team']}]"
        )
    print(f"\nSaved {len(assignments)} assignments to DB.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    args = parser.parse_args()
    main(season=args.season, week=args.week)
