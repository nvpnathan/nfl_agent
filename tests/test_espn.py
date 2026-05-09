import pytest
from src.data.espn import parse_game, parse_box_score, _parse_split, _parse_possession


def _make_event(espn_id="401220225", home_abbr="HOU", away_abbr="BUF",
                home_score="20", away_score="13", status="STATUS_FINAL",
                season_year=2019, season_type=3, week=1):
    return {
        "id": espn_id,
        "date": "2020-01-04T18:05Z",
        "season": {"year": season_year, "type": season_type},
        "week": {"number": week},
        "status": {"type": {"name": status}},
        "competitions": [{
            "neutralSite": False,
            "attendance": 70000,
            "venue": {"fullName": "NRG Stadium", "indoor": True},
            "competitors": [
                {
                    "homeAway": "home",
                    "score": home_score,
                    "team": {"abbreviation": home_abbr, "id": "34"},
                },
                {
                    "homeAway": "away",
                    "score": away_score,
                    "team": {"abbreviation": away_abbr, "id": "2"},
                },
            ],
        }],
    }


def test_parse_split_made_att():
    assert _parse_split("7-14") == (7, 14)
    assert _parse_split("0-0") == (0, 0)
    assert _parse_split("3-4") == (3, 4)


def test_parse_possession_to_seconds():
    assert _parse_possession("32:14") == 32 * 60 + 14
    assert _parse_possession("0:00") == 0
    assert _parse_possession("28:30") == 28 * 60 + 30


def test_parse_game_extracts_teams():
    event = _make_event()
    game = parse_game(event)
    assert game["home_team"] == "HOU"
    assert game["away_team"] == "BUF"
    assert game["espn_id"] == "401220225"
    assert game["season"] == 2019
    assert isinstance(game["is_indoor"], int)
    assert isinstance(game["is_neutral"], int)


def test_parse_game_completed_sets_home_win():
    event = _make_event(home_score="20", away_score="13")
    game = parse_game(event)
    assert game["home_win"] == 1
    assert game["home_score"] == 20
    assert game["away_score"] == 13


def test_parse_game_postseason_type():
    event = _make_event(season_type=3, week=1)
    game = parse_game(event)
    assert game["game_type"] == "wildcard"


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
