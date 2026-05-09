import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from src.model.train import train_model, load_model
from src.model.predict import predict_game_prob, predict_week
from src.features.builder import FEATURE_COLS


def make_fake_training_data(n=200) -> pd.DataFrame:
    np.random.seed(42)
    df = pd.DataFrame({col: np.random.random(n) for col in FEATURE_COLS})
    df["home_spread"] = np.random.uniform(-14, 14, n)
    df["game_total"] = np.random.uniform(38, 58, n)
    df["home_win"] = (-df["home_spread"] + np.random.normal(0, 3, n) > 0).astype(int)
    return df


def test_train_model_saves_and_loads(tmp_path):
    df = make_fake_training_data()
    model_path = str(tmp_path / "model.joblib")
    train_model(df, model_path)
    assert os.path.exists(model_path)
    assert load_model(model_path) is not None


def test_predict_game_prob_returns_valid_probability(tmp_path):
    df = make_fake_training_data()
    model_path = str(tmp_path / "model.joblib")
    train_model(df, model_path)
    features = {col: 0.5 for col in FEATURE_COLS}
    features["home_spread"] = -3.0
    features["game_total"] = 45.5
    prob = predict_game_prob(model_path, features)
    assert 0.0 <= prob <= 1.0


def test_predict_favors_spread_favorite(tmp_path):
    df = make_fake_training_data(n=500)
    model_path = str(tmp_path / "model.joblib")
    train_model(df, model_path)
    base = {col: 0.5 for col in FEATURE_COLS}
    base["home_qb_active"] = 1
    base["away_qb_active"] = 1
    base["game_total"] = 45.5
    heavy_favorite = {**base, "home_spread": -10.0}
    heavy_underdog = {**base, "home_spread": 10.0}
    assert predict_game_prob(model_path, heavy_favorite) > predict_game_prob(model_path, heavy_underdog)


def test_predict_week_returns_required_keys(tmp_path):
    df = make_fake_training_data()
    model_path = str(tmp_path / "model.joblib")
    train_model(df, model_path)
    features = {col: 0.5 for col in FEATURE_COLS}
    features["home_spread"] = -3.0
    features["game_total"] = 45.5
    games = [{
        "espn_id": "401220225",
        "home_team": "KC",
        "away_team": "BAL",
        "features": features,
    }]
    results = predict_week(model_path, games)
    assert len(results) == 1
    r = results[0]
    assert "home_win_prob" in r
    assert "predicted_winner" in r
    assert "win_probability" in r
    assert 0.0 <= r["home_win_prob"] <= 1.0
    assert r["predicted_winner"] in ("KC", "BAL")


from src.model.evaluate import compute_week_metrics, baseline_accuracy


def test_baseline_accuracy_always_picks_favorite():
    predictions = [
        {"home_win_prob": 0.70, "home_win": 1},
        {"home_win_prob": 0.40, "home_win": 1},
        {"home_win_prob": 0.65, "home_win": 0},
    ]
    acc = baseline_accuracy(predictions)
    assert abs(acc - 1 / 3) < 0.01


def test_compute_week_metrics():
    predictions = [
        {"home_win_prob": 0.70, "home_win": 1, "confidence_points": 3},
        {"home_win_prob": 0.60, "home_win": 0, "confidence_points": 2},
        {"home_win_prob": 0.55, "home_win": 1, "confidence_points": 1},
    ]
    metrics = compute_week_metrics(predictions)
    assert metrics["actual_points"] == 4
    assert metrics["accuracy"] == pytest.approx(2 / 3)
