import httpx
import requests

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

_POSTSEASON_WEEK_TO_TYPE = {
    1: "wildcard",
    2: "divisional",
    3: "conference",
    4: "superbowl",
    5: "superbowl",  # ESPN sometimes uses week 5
}


def _get(path: str, params: dict = None) -> dict:
    url = f"{ESPN_BASE}/{path}"
    with httpx.Client(timeout=30) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def fetch_scoreboard(season: int, week: int, season_type: int = 2) -> list[dict]:
    """Return list of event dicts. season_type: 2=regular, 3=postseason."""
    data = _get("scoreboard", params={"dates": season, "week": week, "seasontype": season_type})
    return data.get("events", [])


def fetch_game_summary(espn_id: str) -> dict:
    """Return full game summary including boxscore."""
    return _get("summary", params={"event": espn_id})


def fetch_team_injuries(team_id: str) -> list[dict]:
    """Return current raw injury list for a team."""
    data = _get(f"teams/{team_id}/injuries")
    return data.get("injuries", [])


def fetch_team_depth_chart(team_id: str) -> dict:
    """Return raw depth chart response for a team."""
    return _get(f"teams/{team_id}/depthcharts")


def _parse_split(value: str) -> tuple[int, int]:
    """Parse 'made-att' string like '7-14' → (7, 14)."""
    parts = value.split("-")
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return 0, 0
    return 0, 0


def _parse_possession(value: str) -> int:
    """Parse 'MM:SS' to total seconds."""
    parts = value.split(":")
    if len(parts) == 2:
        try:
            return int(parts[0]) * 60 + int(parts[1])
        except ValueError:
            return 0
    return 0


def parse_game(event: dict) -> dict:
    """Normalize an ESPN scoreboard event to a flat dict for DB insertion."""
    comp = event.get("competitions", [{}])[0]
    competitors = comp.get("competitors", [])
    home = next((c for c in competitors if c.get("homeAway") == "home"), {})
    away = next((c for c in competitors if c.get("homeAway") == "away"), {})

    season_info = event.get("season", {})
    season_type = season_info.get("type", 2)
    week_num = event.get("week", {}).get("number", 0)

    if season_type == 3:
        game_type = _POSTSEASON_WEEK_TO_TYPE.get(week_num, "wildcard")
    else:
        game_type = "regular"

    venue = comp.get("venue", {})
    status_name = event.get("status", {}).get("type", {}).get("name", "")
    is_final = "FINAL" in status_name.upper()

    home_score_raw = home.get("score")
    away_score_raw = away.get("score")

    try:
        home_score = int(float(home_score_raw)) if home_score_raw is not None else None
        away_score = int(float(away_score_raw)) if away_score_raw is not None else None
    except (ValueError, TypeError):
        home_score = away_score = None

    if is_final and home_score is not None and away_score is not None:
        home_win = int(home_score > away_score)
    else:
        home_win = None

    home_team = home.get("team", {})
    away_team = away.get("team", {})

    return {
        "espn_id": str(event["id"]),
        "season": season_info.get("year"),
        "week": week_num,
        "game_type": game_type,
        "home_team": home_team.get("abbreviation", ""),
        "away_team": away_team.get("abbreviation", ""),
        "home_espn_id": str(home_team.get("id", "")),
        "away_espn_id": str(away_team.get("id", "")),
        "game_date": event.get("date", ""),
        "venue": venue.get("fullName", ""),
        "is_indoor": int(venue.get("indoor", False)),
        "is_neutral": int(comp.get("neutralSite", False)),
        "attendance": comp.get("attendance"),
        "home_score": home_score,
        "away_score": away_score,
        "home_win": home_win,
    }


def parse_box_score(summary: dict, espn_id: str) -> list[dict]:
    """Parse ESPN summary boxscore into per-team stat dicts."""
    teams = summary.get("boxscore", {}).get("teams", [])
    results = []
    for team_data in teams:
        team = team_data.get("team", {})
        is_home = int(team_data.get("homeAway", "away") == "home")
        stats = {s["name"]: s.get("displayValue", "0") for s in team_data.get("statistics", [])}

        third_made, third_att = _parse_split(stats.get("thirdDownEff", "0-0"))
        rz_made, rz_att = _parse_split(stats.get("redZoneAtts", "0-0"))
        sacks_taken, _ = _parse_split(stats.get("sacksYardsLost", "0-0"))

        def _int(key: str) -> int:
            try:
                return int(stats.get(key, 0) or 0)
            except (ValueError, TypeError):
                return 0

        results.append({
            "espn_id": espn_id,
            "team": team.get("abbreviation", ""),
            "is_home": is_home,
            "total_yards": _int("totalYards"),
            "pass_yards": _int("netPassingYards"),
            "rush_yards": _int("rushingYards"),
            "turnovers": _int("turnovers"),
            "first_downs": _int("firstDowns"),
            "third_down_att": third_att,
            "third_down_made": third_made,
            "red_zone_att": rz_att,
            "red_zone_made": rz_made,
            "possession_secs": _parse_possession(stats.get("possessionTime", "0:00")),
            "sacks_taken": sacks_taken,
        })
    return results


def parse_game_injuries(summary: dict, season: int, week: int) -> list[dict]:
    """Extract game-day injury data from a summary (for historical backfill)."""
    injuries = []
    for section in summary.get("injuries", []):
        for item in section.get("injuries", []):
            athlete = item.get("athlete", {})
            position = athlete.get("position", {}).get("abbreviation", "")
            team = athlete.get("team", {}).get("abbreviation", "")
            if not team or not athlete.get("id"):
                continue
            injuries.append({
                "season": season,
                "week": week,
                "team": team,
                "athlete_id": str(athlete["id"]),
                "athlete_name": athlete.get("displayName", ""),
                "position": position,
                "status": item.get("status", ""),
                "is_qb": int(position == "QB"),
            })
    return injuries


def parse_depth_chart_qbs(depth_chart: dict, team_abbr: str, season: int, week: int) -> list[dict]:
    """Extract QB depth chart entries (rank 1 = starter)."""
    entries = []
    for item in depth_chart.get("items", []):
        positions = item.get("positions", {})
        qb_data = positions.get("QB", {})
        for athlete_entry in qb_data.get("athletes", []):
            athlete = athlete_entry.get("athlete", {})
            entries.append({
                "season": season,
                "week": week,
                "team": team_abbr,
                "athlete_id": str(athlete.get("id", "")),
                "athlete_name": athlete.get("displayName", ""),
                "rank": int(athlete_entry.get("rank", 99)),
            })
    return entries


def fetch_game_odds(espn_id: str) -> dict | None:
    """Fetch Bet365 pre-match spread and total from ESPN. Returns None if unavailable."""
    url = (
        f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
        f"/events/{espn_id}/competitions/{espn_id}/odds"
    )
    resp = requests.get(url, timeout=10)
    if not resp.ok:
        return None
    data = resp.json()
    bet365 = next(
        (i for i in data.get("items", []) if i.get("provider", {}).get("id") == "2000"),
        None,
    )
    if not bet365:
        return None
    team_odds = bet365.get("bettingOdds", {}).get("teamOdds", {})
    spread = team_odds.get("preMatchSpreadHandicapHome", {}).get("value")
    total = team_odds.get("preMatchTotalHandicap", {}).get("value")
    if spread is None or total is None:
        return None
    return {
        "espn_id": espn_id,
        "home_spread": float(spread),
        "game_total": float(total),
        "home_moneyline": team_odds.get("preMatchMoneyLineHome", {}).get("value"),
        "away_moneyline": team_odds.get("preMatchMoneyLineAway", {}).get("value"),
    }
