#!/usr/bin/env python3
"""Walk-forward backtest: train on 2019-N, validate on N+1 for N in [2021, 2022, 2023].

Reports accuracy, Brier score, expected confidence points, and actual confidence points
per fold and averaged across folds. Expected points and actual points are the key metrics
for the confidence pool goal.
"""
import argparse
import tempfile
from pathlib import Path

import yaml
import pandas as pd
from src.features.builder import build_training_dataset
from src.model.train import train_model
from src.model.evaluate import run_season_backtest

FOLDS_ALL = [
    (list(range(2018, 2022)), 2022),
    (list(range(2018, 2023)), 2023),
    (list(range(2018, 2024)), 2024),
    (list(range(2018, 2025)), 2025),
]

FOLDS_RECENT = [
    (list(range(2021, 2022)), 2022),
    (list(range(2021, 2023)), 2023),
    (list(range(2021, 2024)), 2024),
    (list(range(2021, 2025)), 2025),
]


def _run_folds(folds, db_path, model_path, strategy_label: str) -> pd.DataFrame:
    all_results = []
    for train_seasons, val_season in folds:
        print(f"\nFold [{strategy_label}]: train={train_seasons} → validate={val_season}")
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
    return df


def main(
    write_production_artifact: bool = False,
    strategy: str = "all",
) -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    production_model_path = config["model"]["path"]
    db_path = config["db"]["path"]

    if strategy == "recent":
        folds = FOLDS_RECENT
    else:
        folds = FOLDS_ALL

    tmpdir_obj = None
    if write_production_artifact:
        tmp_dir = None
        print(f"Using production artifact path: {production_model_path}")
    else:
        tmpdir_obj = tempfile.TemporaryDirectory(prefix="nfl_backtest_")
        tmp_dir = Path(tmpdir_obj.name)
        print(f"Using temp artifact directory: {tmp_dir}")

    try:
        df = _run_folds(folds, db_path, str(tmp_dir / "model_fold_{val_season}.joblib"),
                        strategy_label=strategy)

        print(f"\n=== Walk-forward summary [{strategy}] ===")
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
    finally:
        if tmpdir_obj is not None:
            tmpdir_obj.cleanup()


def main_compare(db_path: str) -> None:
    """Run both strategies side by side and print a comparison table."""
    tmpdir_obj = tempfile.TemporaryDirectory(prefix="nfl_backtest_compare_")
    tmp_dir = Path(tmpdir_obj.name)

    try:
        df_all = _run_folds(FOLDS_ALL, db_path, str(tmp_dir / "model_{val_season}.joblib"),
                            strategy_label="all")
        df_recent = _run_folds(FOLDS_RECENT, db_path, str(tmp_dir / "model_{val_season}.joblib"),
                               strategy_label="recent")

        def summary_stats(df):
            return {
                "accuracy": df["accuracy"].mean(),
                "brier_score": df["brier_score"].mean(),
                "expected_points": df["expected_points"].mean(),
                "actual_points": df["actual_points"].mean(),
            }

        all_stats = summary_stats(df_all)
        recent_stats = summary_stats(df_recent)

        print(f"\n{'Metric':<20s} {'All Seasons':>14s} {'Recent (2021+)':>16s}")
        print("-" * 54)
        for key in ["accuracy", "brier_score", "expected_points", "actual_points"]:
            label = key.replace("_", " ").title()
            print(f"{label:<20s} {all_stats[key]:>14.3f} {recent_stats[key]:>16.3f}")

        print(f"\n=== Per-fold comparison ===")
        per_fold = pd.DataFrame({
            "val_season": [2022, 2023, 2024, 2025],
            "all_accuracy": df_all.groupby("fold_val_season")["accuracy"].mean().values,
            "recent_accuracy": df_recent.groupby("fold_val_season")["accuracy"].mean().values,
            "all_expected_pts": df_all.groupby("fold_val_season")["expected_points"].mean().values,
            "recent_expected_pts": df_recent.groupby("fold_val_season")["expected_points"].mean().values,
        })
        print(per_fold.to_string(index=False))
    finally:
        tmpdir_obj.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run walk-forward backtest. Supports 'all' (2018+) and 'recent' (2021+)."
    )
    parser.add_argument(
        "--write-production-artifact",
        action="store_true",
        help="Write fold artifacts to config.model.path (legacy behavior, overwrites each fold).",
    )
    parser.add_argument(
        "--strategy",
        choices=["all", "recent"],
        default="all",
        help="Training strategy: 'all' uses 2018+, 'recent' uses 2021+ (default: all).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both strategies side by side and print a comparison table.",
    )
    args = parser.parse_args()

    if args.compare:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        main_compare(config["db"]["path"])
    else:
        main(write_production_artifact=args.write_production_artifact, strategy=args.strategy)
