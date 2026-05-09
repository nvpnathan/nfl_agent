import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
from src.features.builder import FEATURE_COLS

MODEL_VERSION = "xgb_v1"

def train_model(df: pd.DataFrame, model_path: str) -> dict:
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = df["home_win"]
    medians = df[FEATURE_COLS].median().to_dict()

    base = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model = CalibratedClassifierCV(base, cv=5, method="sigmoid")
    model.fit(X, y)

    cv_scores = cross_val_score(base, X, y, cv=5, scoring="accuracy")
    joblib.dump({"model": model, "version": MODEL_VERSION, "features": FEATURE_COLS, "medians": medians}, model_path)

    return {
        "model_version": MODEL_VERSION,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "n_samples": len(df),
    }

def load_model(model_path: str):
    return joblib.load(model_path)
