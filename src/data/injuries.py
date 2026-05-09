"""ESPN-backed injury fetcher. Replaces Sleeper API integration."""
from src.data.espn import fetch_team_injuries

OUT_STATUSES = {"Out", "Doubtful", "IR", "PUP-R"}


def fetch_espn_injuries(team_id: str, team_abbr: str, season: int, week: int) -> list[dict]:
    """Fetch current injury report for a team from ESPN."""
    raw = fetch_team_injuries(team_id)
    result = []
    for item in raw:
        athlete = item.get("athlete", {})
        position = athlete.get("position", {}).get("abbreviation", "")
        result.append({
            "season": season,
            "week": week,
            "team": team_abbr,
            "athlete_id": str(athlete.get("id", "")),
            "athlete_name": athlete.get("displayName", ""),
            "position": position,
            "status": item.get("status", ""),
            "is_qb": int(position == "QB"),
        })
    return result


def is_qb_out(injuries: list[dict]) -> bool:
    return any(
        i["is_qb"] and i.get("status") in OUT_STATUSES
        for i in injuries
    )
