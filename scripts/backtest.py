#!/usr/bin/env python3
"""Backtest the model on validation seasons and print summary."""
import yaml
import pandas as pd
from src.features.builder import build_training_dataset
from src.model.train import train_model
from src.model.evaluate import run_season_backtest


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    train_seasons = config["model"]["train_seasons"]
    val_seasons = config["model"]["val_seasons"]
    model_path = config["model"]["path"]
    db_path = config["db"]["path"]

    print(f"Building training dataset for seasons {train_seasons}...")
    train_df = build_training_dataset(seasons=train_seasons)
    print(f"Training on {len(train_df)} games...")
    metrics = train_model(train_df, model_path)
    print(f"Training complete: CV accuracy={metrics['cv_accuracy_mean']:.3f}")

    print(f"\nRunning backtest on {val_seasons}...")
    results = run_season_backtest(model_path, val_seasons, db_path)
    df = pd.DataFrame(results)
    print(f"\nBacktest Results:")
    print(f"  Avg accuracy:      {df['accuracy'].mean():.3f}")
    print(f"  Baseline accuracy: {df['baseline_accuracy'].mean():.3f}")
    print(f"  Avg actual pts/wk: {df['actual_points'].mean():.1f}")
    print(f"  Avg expected pts:  {df['expected_points'].mean():.1f}")
    print(f"  Avg Brier score:   {df['brier_score'].mean():.4f}")


if __name__ == "__main__":
    main()
