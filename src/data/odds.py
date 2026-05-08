import requests
import yaml
from typing import Optional


def moneyline_to_prob(ml: int) -> float:
    """Convert moneyline odds to implied probability."""
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    return 100 / (ml + 100)


def parse_odds_response(games: list[dict]) -> list[dict]:
    """
    Parse The Odds API response and remove vig (vigorish/overround).

    Returns games with home_win_prob and away_win_prob summing to ~1.0.
    """
    results = []
    for game in games:
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue
        h2h = next(
            (m for m in bookmakers[0].get("markets", []) if m["key"] == "h2h"),
            None
        )
        if not h2h:
            continue
        outcomes = {o["name"]: o["price"] for o in h2h["outcomes"]}
        home = game["home_team"]
        away = game["away_team"]
        if home not in outcomes or away not in outcomes:
            continue
        home_implied = moneyline_to_prob(outcomes[home])
        away_implied = moneyline_to_prob(outcomes[away])
        total = home_implied + away_implied
        results.append({
            "odds_game_id": game["id"],
            "home_team": home,
            "away_team": away,
            "home_win_prob": home_implied / total,
            "away_win_prob": away_implied / total,
            "home_moneyline": outcomes[home],
            "away_moneyline": outcomes[away],
        })
    return results


def fetch_current_odds(config_path: str = "config.yaml") -> list[dict]:
    """Fetch current odds from The Odds API."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    api_key = config["odds_api"]["key"]
    sport = config["odds_api"]["sport"]
    regions = config["odds_api"]["regions"]
    markets = config["odds_api"]["markets"]
    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
        f"?apiKey={api_key}&regions={regions}&markets={markets}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return parse_odds_response(resp.json())
