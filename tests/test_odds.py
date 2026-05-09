from unittest.mock import patch, MagicMock
from src.data.espn import fetch_game_odds


def _make_espn_response(spread, total, home_ml, away_ml):
    return {
        "items": [{
            "provider": {"id": "2000", "name": "Bet 365"},
            "bettingOdds": {
                "teamOdds": {
                    "preMatchSpreadHandicapHome": {"value": spread},
                    "preMatchTotalHandicap": {"value": total},
                    "preMatchMoneyLineHome": {"value": home_ml},
                    "preMatchMoneyLineAway": {"value": away_ml},
                }
            }
        }]
    }


def test_fetch_game_odds_parses_bet365_fields():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = _make_espn_response("-2.5", "46.5", "4/6", "5/4")
    with patch("src.data.espn.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.return_value = mock_resp
        result = fetch_game_odds("401671789")
    assert result is not None
    assert result["espn_id"] == "401671789"
    assert result["home_spread"] == -2.5
    assert result["game_total"] == 46.5
    assert result["home_moneyline"] == "4/6"
    assert result["away_moneyline"] == "5/4"


def test_fetch_game_odds_returns_none_when_no_bet365():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "items": [{"provider": {"id": "58", "name": "ESPN BET"}, "bettingOdds": {}}]
    }
    with patch("src.data.espn.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.return_value = mock_resp
        result = fetch_game_odds("401671789")
    assert result is None


def test_fetch_game_odds_returns_none_when_spread_missing():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "items": [{
            "provider": {"id": "2000", "name": "Bet 365"},
            "bettingOdds": {
                "teamOdds": {
                    "preMatchSpreadHandicapHome": {"value": None},
                    "preMatchTotalHandicap": {"value": "46.5"},
                    "preMatchMoneyLineHome": {"value": "4/6"},
                    "preMatchMoneyLineAway": {"value": "5/4"},
                }
            }
        }]
    }
    with patch("src.data.espn.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.return_value = mock_resp
        result = fetch_game_odds("401671789")
    assert result is None


def test_fetch_game_odds_returns_none_on_http_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    with patch("src.data.espn.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.get.return_value = mock_resp
        result = fetch_game_odds("401671789")
    assert result is None
