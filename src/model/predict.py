import pandas as pd
from src.model.train import load_model
from src.features.builder import FEATURE_COLS


def predict_game_prob(model_path: str, features: dict) -> float:
    artifact = load_model(model_path)
    model = artifact["model"]
    medians = artifact.get("medians", {})
    X = pd.DataFrame([features])[FEATURE_COLS].fillna(medians).fillna(0.5)
    return float(model.predict_proba(X)[0][1])


def predict_week(model_path: str, games_features: list[dict]) -> list[dict]:
    artifact = load_model(model_path)
    model = artifact["model"]
    medians = artifact.get("medians", {})
    results = []
    for game in games_features:
        X = pd.DataFrame([game["features"]])[FEATURE_COLS].fillna(medians).fillna(0.5)
        home_prob = float(model.predict_proba(X)[0][1])
        winner = game["home_team"] if home_prob >= 0.5 else game["away_team"]
        results.append({
            **game,
            "home_win_prob": home_prob,
            "away_win_prob": 1.0 - home_prob,
            "predicted_winner": winner,
            "win_probability": max(home_prob, 1.0 - home_prob),
        })
    return results
