import pytest
from src.optimizer.confidence import assign_confidence_points, get_point_range

def test_assign_confidence_highest_prob_gets_most_points():
    games = [
        {"game_id": "g1", "predicted_winner": "KC", "win_probability": 0.80},
        {"game_id": "g2", "predicted_winner": "SF", "win_probability": 0.65},
        {"game_id": "g3", "predicted_winner": "DAL", "win_probability": 0.55},
    ]
    result = assign_confidence_points(games, (1, 3))
    by_id = {r["game_id"]: r for r in result}
    assert by_id["g1"]["confidence_points"] == 3
    assert by_id["g2"]["confidence_points"] == 2
    assert by_id["g3"]["confidence_points"] == 1

def test_all_points_used_exactly_once():
    games = [
        {"game_id": f"g{i}", "predicted_winner": "X", "win_probability": 0.5 + i*0.01}
        for i in range(16)
    ]
    result = assign_confidence_points(games, (1, 16))
    points = sorted(r["confidence_points"] for r in result)
    assert points == list(range(1, 17))

def test_uncertainty_flag_within_threshold():
    games = [
        {"game_id": "g1", "predicted_winner": "KC", "win_probability": 0.72},
        {"game_id": "g2", "predicted_winner": "SF", "win_probability": 0.71},  # within 3%
        {"game_id": "g3", "predicted_winner": "DAL", "win_probability": 0.55},
    ]
    result = assign_confidence_points(games, (1, 3), uncertainty_threshold=0.03)
    by_id = {r["game_id"]: r for r in result}
    assert by_id["g1"]["is_uncertain"] is True
    assert by_id["g2"]["is_uncertain"] is True
    assert by_id["g3"]["is_uncertain"] is False

def test_get_point_range_regular_season():
    assert get_point_range(n_games=16, game_type="regular") == (1, 16)
    assert get_point_range(n_games=15, game_type="regular") == (1, 15)

def test_get_point_range_playoff():
    assert get_point_range(n_games=6, game_type="wildcard") == (1, 6)

def test_underfull_week_assigns_one_through_n():
    games = [
        {"game_id": f"g{i}", "predicted_winner": "X", "win_probability": 0.5 + i*0.01}
        for i in range(14)
    ]
    result = assign_confidence_points(games, (1, 16))
    points = sorted(r["confidence_points"] for r in result)
    assert points == list(range(1, 15))  # 1..14, not 3..16
