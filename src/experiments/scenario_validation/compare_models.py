#!/usr/bin/env python3
"""
Benchmark the legacy L-bracket PINN against the new multi-geometry PINN.
Outputs per-geometry/load metrics and a comparison plot.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data/results/multi_geom_training"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path(__file__).resolve().parent / "figures" / "multi_geom"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MULTI_MODEL_DIR = PROJECT_ROOT / "src/approach_a_pinn/artifacts_multi_geom"
LEGACY_MODEL_DIR = PROJECT_ROOT / "src/approach_a_pinn/artifacts"

SCREW_COLS = ["s1_x", "s1_y", "s2_x", "s2_y", "s3_x", "s3_y"]


class SurrogateModel(torch.nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_multi_model():
    model_path = MULTI_MODEL_DIR / "pinn_multi_geom.pth"
    stats_path = MULTI_MODEL_DIR / "norm_stats_multi_geom.npz"
    meta_path = MULTI_MODEL_DIR / "dataset_metadata.json"
    if not model_path.exists():
        raise SystemExit("Multi-geometry model not found. Train it first.")

    meta = json.loads(meta_path.read_text())
    hidden = meta.get("hidden_size", 96)
    model = SurrogateModel(meta["input_dim"], hidden)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    stats = np.load(stats_path)
    return model, stats, meta


def load_legacy_model():
    model_path = LEGACY_MODEL_DIR / "pinn_model.pth"
    stats_path = LEGACY_MODEL_DIR / "norm_stats.npz"
    if not model_path.exists():
        print("⚠️ Legacy model not found; skipping legacy comparison.")
        return None, None
    model = SurrogateModel(4, 64)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    stats = np.load(stats_path)
    return model, stats


def load_dataset(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(path)
        df["source_file"] = path.name
        for col in SCREW_COLS:
            if col not in df.columns:
                df[col] = 0.0
        frames.append(df)
    if not frames:
        raise SystemExit(f"No CSV files found in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def encode_multi_features(df: pd.DataFrame, meta: dict):
    geom_map = {name: idx for idx, name in enumerate(meta["geom_names"])}
    load_map = {name: idx for idx, name in enumerate(meta["load_cases"])}

    screw = df[SCREW_COLS].values.astype(np.float32)
    geom_one_hot = np.eye(len(geom_map))[df["geometry"].map(geom_map)]
    load_one_hot = np.eye(len(load_map))[df["load_case"].map(load_map)]
    return np.hstack([screw, geom_one_hot, load_one_hot])


def predict_multi(model, stats, meta, df):
    X = encode_multi_features(df, meta)
    X_norm = (X - stats["X_mean"]) / stats["X_std"]
    start = time.perf_counter()
    with torch.no_grad():
        preds = model(torch.tensor(X_norm, dtype=torch.float32)).numpy()
    elapsed = time.perf_counter() - start
    y = preds * stats["y_std"] + stats["y_mean"]
    return y.flatten(), elapsed / len(df)


def predict_legacy(model, stats, df):
    subset = df[(df["s3_x"] == 0) & (df["s3_y"] == 0)].copy()
    if subset.empty:
        return None, None, None
    X = subset[SCREW_COLS[:4]].values.astype(np.float32)
    X_norm = (X - stats["X_mean"]) / stats["X_std"]
    start = time.perf_counter()
    with torch.no_grad():
        preds = model(torch.tensor(X_norm, dtype=torch.float32)).numpy()
    elapsed = time.perf_counter() - start
    y = preds * stats["y_mean"] + stats["y_std"]
    return subset.index.values, y.flatten(), elapsed / len(subset)


def compute_metrics(truth: np.ndarray, preds: np.ndarray):
    mae = np.mean(np.abs(preds - truth))
    mape = np.mean(np.abs(preds - truth) / np.maximum(np.abs(truth), 1e-6)) * 100
    return mae, mape


def main():
    parser = argparse.ArgumentParser(description="Benchmark legacy vs multi-geometry PINNs.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    df = load_dataset(args.data_dir)
    multi_model, multi_stats, meta = load_multi_model()
    legacy_model, legacy_stats = load_legacy_model()

    multi_preds, multi_latency = predict_multi(multi_model, multi_stats, meta, df)
    df["multi_pred"] = multi_preds
    df["multi_latency_ms"] = multi_latency * 1000.0

    legacy_idx = None
    if legacy_model is not None:
        legacy_idx, legacy_preds, legacy_latency = predict_legacy(legacy_model, legacy_stats, df)
        if legacy_idx is not None:
            df.loc[legacy_idx, "legacy_pred"] = legacy_preds
            df.loc[legacy_idx, "legacy_latency_ms"] = legacy_latency * 1000.0

    records = []
    for (geom, load_case), group in df.groupby(["geometry", "load_case"]):
        multi_mae, multi_mape = compute_metrics(group["compliance"].values, group["multi_pred"].values)
        record = {
            "geometry": geom,
            "load_case": load_case,
            "num_samples": len(group),
            "multi_mae": multi_mae,
            "multi_mape_pct": multi_mape,
            "multi_latency_ms": group["multi_latency_ms"].iloc[0],
        }
        legacy_group = group["legacy_pred"].dropna()
        if not legacy_group.empty:
            idxs = legacy_group.index
            truth = group.loc[idxs, "compliance"].values
            preds = legacy_group.values
            legacy_mae, legacy_mape = compute_metrics(truth, preds)
            record["legacy_mae"] = legacy_mae
            record["legacy_mape_pct"] = legacy_mape
            record["legacy_latency_ms"] = group["legacy_latency_ms"].iloc[0]
        records.append(record)

    metrics_df = pd.DataFrame(records)
    metrics_path = RESULTS_DIR / "multi_geom_model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✅ Metrics saved to {metrics_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = metrics_df["geometry"] + " / " + metrics_df["load_case"]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, metrics_df["multi_mae"], width, label="Multi-Geometry PINN")
    if "legacy_mae" in metrics_df:
        ax.bar(x + width/2, metrics_df["legacy_mae"].fillna(0), width, label="Legacy PINN")
    ax.set_ylabel("MAE (J)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Model MAE per Geometry/Load")
    ax.legend()
    fig.tight_layout()
    plot_path = FIG_DIR / "model_mae_comparison.png"
    fig.savefig(plot_path, dpi=200)
    print(f"✅ Plot saved to {plot_path}")


if __name__ == "__main__":
    main()

