import json
from pathlib import Path
import pytest
from src.data.espn import parse_game, parse_box_score, _parse_split, _parse_possession


def load_season_fixture():
    return json.loads(Path("nfl_season_2020.json").read_text())


def test_parse_split_made_att():
    assert _parse_split("7-14") == (7, 14)
    assert _parse_split("0-0") == (0, 0)
    assert _parse_split("3-4") == (3, 4)


def test_parse_possession_to_seconds():
    assert _parse_possession("32:14") == 32 * 60 + 14
    assert _parse_possession("0:00") == 0
    assert _parse_possession("28:30") == 28 * 60 + 30


def test_parse_game_extracts_teams():
    data = load_season_fixture()
    event = data["events"][0]
    game = parse_game(event)
    assert game["home_team"] == "HOU"
    assert game["away_team"] == "BUF"
    assert game["espn_id"] == event["id"]
    assert game["season"] == 2019  # ESPN year for Jan 2020 playoff game
    assert isinstance(game["is_indoor"], int)
    assert isinstance(game["is_neutral"], int)


def test_parse_game_completed_sets_home_win():
    data = load_season_fixture()
    # First event is a completed playoff game (STATUS_FINAL)
    event = data["events"][0]
    game = parse_game(event)
    assert game["home_win"] in (0, 1)
    assert game["home_score"] is not None
    assert game["away_score"] is not None


def test_parse_game_postseason_type():
    data = load_season_fixture()
    # All events in season fixture are post-season (type=3)
    event = data["events"][0]
    game = parse_game(event)
    assert game["game_type"] in ("wildcard", "divisional", "conference", "superbowl")


def test_parse_box_score_returns_two_teams():
    summary = {
        "boxscore": {
            "teams": [
                {
                    "team": {"abbreviation": "HOU"},
                    "homeAway": "home",
                    "statistics": [
                        {"name": "totalYards", "displayValue": "300"},
                        {"name": "netPassingYards", "displayValue": "200"},
                        {"name": "rushingYards", "displayValue": "100"},
                        {"name": "turnovers", "displayValue": "2"},
                        {"name": "firstDowns", "displayValue": "18"},
                        {"name": "thirdDownEff", "displayValue": "5-13"},
                        {"name": "redZoneAtts", "displayValue": "2-3"},
                        {"name": "possessionTime", "displayValue": "28:30"},
                        {"name": "sacksYardsLost", "displayValue": "3-21"},
                    ],
                },
                {
                    "team": {"abbreviation": "BUF"},
                    "homeAway": "away",
                    "statistics": [
                        {"name": "totalYards", "displayValue": "350"},
                        {"name": "turnovers", "displayValue": "1"},
                        {"name": "thirdDownEff", "displayValue": "7-12"},
                        {"name": "redZoneAtts", "displayValue": "3-4"},
                        {"name": "possessionTime", "displayValue": "31:30"},
                        {"name": "sacksYardsLost", "displayValue": "1-7"},
                    ],
                },
            ]
        }
    }
    results = parse_box_score(summary, "401220225")
    assert len(results) == 2
    hou = next(r for r in results if r["team"] == "HOU")
    assert hou["total_yards"] == 300
    assert hou["turnovers"] == 2
    assert hou["third_down_att"] == 13
    assert hou["third_down_made"] == 5
    assert hou["possession_secs"] == 28 * 60 + 30
    assert hou["sacks_taken"] == 3
    assert hou["is_home"] == 1
    buf = next(r for r in results if r["team"] == "BUF")
    assert buf["is_home"] == 0
    assert buf["total_yards"] == 350


def test_parse_box_score_empty_summary():
    results = parse_box_score({}, "999")
    assert results == []
