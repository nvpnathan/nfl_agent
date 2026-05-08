import pytest
import pandas as pd
import numpy as np
import tempfile, os
from src.model.train import train_model, load_model
from src.model.predict import predict_game_prob
from src.features.builder import FEATURE_COLS

def make_fake_training_data(n=200) -> pd.DataFrame:
    np.random.seed(42)
    df = pd.DataFrame({col: np.random.random(n) for col in FEATURE_COLS})
    df["home_win"] = (df["odds_home_win_prob"] + np.random.normal(0, 0.1, n) > 0.5).astype(int)
    return df

def test_train_model_saves_and_loads(tmp_path):
    df = make_fake_training_data()
    model_path = str(tmp_path / "model.joblib")
    train_model(df, model_path)
    assert os.path.exists(model_path)
    model = load_model(model_path)
    assert model is not None

def test_predict_game_prob_returns_valid_probability(tmp_path):
    df = make_fake_training_data()
    model_path = str(tmp_path / "model.joblib")
    train_model(df, model_path)
    features = {col: 0.5 for col in FEATURE_COLS}
    features["odds_home_win_prob"] = 0.65
    prob = predict_game_prob(model_path, features)
    assert 0.0 <= prob <= 1.0

def test_predict_favors_higher_odds_team(tmp_path):
    df = make_fake_training_data(n=500)
    model_path = str(tmp_path / "model.joblib")
    train_model(df, model_path)
    base = {col: 0.5 for col in FEATURE_COLS}
    base["home_qb_out"] = 0
    base["away_qb_out"] = 0
    high_odds = {**base, "odds_home_win_prob": 0.80}
    low_odds = {**base, "odds_home_win_prob": 0.40}
    assert predict_game_prob(model_path, high_odds) > predict_game_prob(model_path, low_odds)

def test_predict_week_returns_required_keys(tmp_path):
    df = make_fake_training_data()
    model_path = str(tmp_path / "model.joblib")
    train_model(df, model_path)
    games = [{
        "game_id": "test_g1",
        "home_team": "KC",
        "away_team": "BAL",
        "features": {col: 0.5 for col in FEATURE_COLS},
    }]
    from src.model.predict import predict_week
    results = predict_week(model_path, games)
    assert len(results) == 1
    r = results[0]
    assert "home_win_prob" in r
    assert "predicted_winner" in r
    assert "win_probability" in r
    assert 0.0 <= r["home_win_prob"] <= 1.0
    assert r["predicted_winner"] in ("KC", "BAL")
