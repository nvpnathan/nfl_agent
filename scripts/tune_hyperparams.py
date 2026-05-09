#!/usr/bin/env python3
"""Walk-forward hyperparameter search for XGBoost.

Evaluates each config across the same 3 folds used in backtest.py.
Optimizes Brier score — calibration is what drives confidence point accuracy.
Skips CalibratedClassifierCV for speed; we tune raw XGBoost params here,
then keep Platt calibration in the final train_model call.

Usage: uv run python scripts/tune_hyperparams.py
"""
import time
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
from src.features.builder import build_training_dataset, FEATURE_COLS

FOLDS = [
    (list(range(2018, 2022)), 2022),
    (list(range(2018, 2023)), 2023),
    (list(range(2018, 2024)), 2024),
]

N_ITER = 50
RNG = np.random.default_rng(42)


def sample_params() -> dict:
    return {
        "n_estimators": int(RNG.choice([200, 300, 500, 700, 1000])),
        "max_depth": int(RNG.integers(3, 6)),
        "learning_rate": float(RNG.choice([0.01, 0.02, 0.04, 0.06, 0.1])),
        "min_child_weight": int(RNG.integers(1, 8)),
        "subsample": round(float(RNG.uniform(0.65, 0.95)), 2),
        "colsample_bytree": round(float(RNG.uniform(0.65, 0.95)), 2),
        "gamma": round(float(RNG.choice([0, 0.05, 0.1, 0.2, 0.5])), 2),
        "reg_alpha": round(float(RNG.choice([0, 0.05, 0.1, 0.5, 1.0])), 2),
        "reg_lambda": round(float(RNG.choice([0.5, 1, 2, 5, 10])), 1),
    }


def eval_config(params: dict, cache: dict[int, pd.DataFrame]) -> dict:
    briers, accs = [], []
    for train_seasons, val_season in FOLDS:
        train_df = pd.concat([cache[s] for s in train_seasons if s in cache], ignore_index=True)
        val_df = cache.get(val_season)
        if train_df.empty or val_df is None or val_df.empty:
            continue

        X_tr = train_df[FEATURE_COLS].fillna(train_df[FEATURE_COLS].median())
        X_va = val_df[FEATURE_COLS].fillna(val_df[FEATURE_COLS].median())
        y_tr, y_va = train_df["home_win"], val_df["home_win"]

        model = XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_va)[:, 1]
        briers.append(brier_score_loss(y_va, probs))
        accs.append(float((probs >= 0.5) == y_va).mean() if False else
                    float(np.mean((probs >= 0.5).astype(int) == y_va.values)))

    return {"brier": float(np.mean(briers)), "accuracy": float(np.mean(accs)), **params}


def main() -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    db_path = config["db"]["path"]

    all_seasons = sorted({s for train, val in FOLDS for s in train} | {val for _, val in FOLDS})
    print(f"Pre-loading season data for {all_seasons}...")
    cache: dict[int, pd.DataFrame] = {}
    for s in all_seasons:
        df = build_training_dataset(db_path, [s])
        if not df.empty:
            cache[s] = df
            print(f"  {s}: {len(df)} games")

    # Baseline (current defaults)
    baseline = {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
        "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8,
        "gamma": 0, "reg_alpha": 0, "reg_lambda": 1,
    }
    print(f"\nEvaluating baseline + {N_ITER} random configs...\n")
    results = []

    b = eval_config(baseline, cache)
    b["label"] = "BASELINE"
    results.append(b)
    print(f"  baseline  brier={b['brier']:.4f}  acc={b['accuracy']:.3f}")

    t0 = time.time()
    for i in range(N_ITER):
        params = sample_params()
        r = eval_config(params, cache)
        r["label"] = f"trial_{i+1:02d}"
        results.append(r)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            best = min(results, key=lambda x: x["brier"])
            print(f"  [{i+1}/{N_ITER}] best so far: brier={best['brier']:.4f}  "
                  f"acc={best['accuracy']:.3f}  ({elapsed:.0f}s)")

    results.sort(key=lambda x: x["brier"])

    print(f"\n{'='*70}")
    print(f"TOP 10 CONFIGS (by Brier score — lower is better)")
    print(f"{'='*70}")
    param_keys = ["n_estimators", "max_depth", "learning_rate", "min_child_weight",
                  "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda"]
    header = f"{'rank':<5} {'brier':<8} {'acc':<7} " + "  ".join(f"{k}" for k in param_keys)
    print(header)
    for i, r in enumerate(results[:10]):
        vals = "  ".join(f"{r[k]}" for k in param_keys)
        marker = " ← BASELINE" if r.get("label") == "BASELINE" else ""
        print(f"  {i+1:<4} {r['brier']:.4f}   {r['accuracy']:.3f}  {vals}{marker}")

    best = results[0]
    if best.get("label") == "BASELINE":
        print("\nBaseline is already optimal — no change needed.")
        return

    print(f"\nBest config (brier={best['brier']:.4f} vs baseline={results[[r['label'] for r in results].index('BASELINE')]['brier']:.4f}):")
    for k in param_keys:
        marker = " *" if best[k] != baseline[k] else ""
        print(f"  {k}: {best[k]}{marker}")

    print("\nTo apply, update src/model/train.py XGBClassifier call with these params,")
    print("then run: uv run python scripts/backtest.py")


if __name__ == "__main__":
    main()
