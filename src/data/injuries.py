import requests
import yaml
from typing import Optional

OUT_STATUSES = {"Out", "Doubtful", "IR", "PUP-R"}
QB_POSITIONS = {"QB"}

def fetch_sleeper_injuries(season: int, week: int,
                            config_path: str = "config.yaml") -> list[dict]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    base = config["sleeper"]["base_url"]
    url = f"{base}/players/nfl"
    # /players/nfl is a current-state snapshot; season/week used for DB tagging only
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    players = resp.json()

    injuries = []
    for player_id, player in players.items():
        status = player.get("injury_status")
        if not status:
            continue
        position = player.get("position", "")
        injuries.append({
            "player_name": player.get("full_name", "Unknown"),
            "position": position,
            "injury_status": status,
            "team": player.get("team", ""),
            "is_qb": int(position in QB_POSITIONS),
            "season": season,
            "week": week,
        })
    return injuries

def parse_sleeper_injuries(raw: list[dict]) -> list[dict]:
    return [p for p in raw if p.get("injury_status") in OUT_STATUSES | {"Questionable"}]

def is_qb_out(injuries: list[dict]) -> bool:
    return any(
        p["is_qb"] and p.get("injury_status") in OUT_STATUSES
        for p in injuries
    )
