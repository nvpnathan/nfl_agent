#!/usr/bin/env python3
"""Walk-forward backtest: train on 2019-N, validate on N+1 for N in [2021, 2022, 2023].

Reports accuracy, Brier score, expected confidence points, and actual confidence points
per fold and averaged across folds. Expected points and actual points are the key metrics
for the confidence pool goal.
"""
import yaml
import pandas as pd
from src.features.builder import build_training_dataset
from src.model.train import train_model
from src.model.evaluate import run_season_backtest

FOLDS = [
    (list(range(2018, 2022)), 2022),
    (list(range(2018, 2023)), 2023),
    (list(range(2018, 2024)), 2024),
    (list(range(2018, 2025)), 2025),
]


def main() -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    model_path = config["model"]["path"]
    db_path = config["db"]["path"]

    all_results = []
    for train_seasons, val_season in FOLDS:
        print(f"\nFold: train={train_seasons} → validate={val_season}")
        train_df = build_training_dataset(db_path=db_path, seasons=train_seasons)
        print(f"  Training on {len(train_df)} games...")
        train_model(train_df, model_path)
        results = run_season_backtest(model_path, [val_season], db_path)
        for r in results:
            r["fold_val_season"] = val_season
        all_results.extend(results)
        fold_df = pd.DataFrame(results)
        print(f"  {val_season}: accuracy={fold_df['accuracy'].mean():.3f} "
              f"brier={fold_df['brier_score'].mean():.4f} "
              f"exp_pts={fold_df['expected_points'].mean():.1f} "
              f"actual_pts={fold_df['actual_points'].mean():.1f}")

    df = pd.DataFrame(all_results)
    print(f"\n=== Walk-forward summary (all folds) ===")
    print(f"  Accuracy:        {df['accuracy'].mean():.3f}  "
          f"(baseline: {df['baseline_accuracy'].mean():.3f})")
    print(f"  Brier score:     {df['brier_score'].mean():.4f}")
    print(f"  Expected pts/wk: {df['expected_points'].mean():.1f}")
    print(f"  Actual pts/wk:   {df['actual_points'].mean():.1f}")

    print(f"\n=== Per-fold summary ===")
    fold_summary = df.groupby("fold_val_season").agg(
        accuracy=("accuracy", "mean"),
        brier_score=("brier_score", "mean"),
        expected_points=("expected_points", "mean"),
        actual_points=("actual_points", "mean"),
    )
    print(fold_summary.to_string())


if __name__ == "__main__":
    main()
