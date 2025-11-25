#!/usr/bin/env python3
"""
Train a PINN surrogate on the combined multi-geometry dataset.
The dataset is expected to live in `data/results/multi_geom_training/`
and contain CSV files emitted by the rollout scripts. Each CSV must include
columns: geometry, load_case, s*_x, s*_y, compliance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = (PROJECT_ROOT / "data/results/multi_geom_training").resolve()
ARTIFACT_DIR = (Path(__file__).resolve().parent / "artifacts_multi_geom").resolve()
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "pinn_multi_geom.pth"
STATS_PATH = ARTIFACT_DIR / "norm_stats_multi_geom.npz"
META_PATH = ARTIFACT_DIR / "dataset_metadata.json"

SCREW_COLS = ["s1_x", "s1_y", "s2_x", "s2_y", "s3_x", "s3_y"]


def load_dataset(files: list[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        df = pd.read_csv(path)
        for col in SCREW_COLS:
            if col not in df.columns:
                df[col] = 0.0
        frames.append(df)
    if not frames:
        raise RuntimeError("No dataset files found.")
    return pd.concat(frames, ignore_index=True)


def encode_features(df: pd.DataFrame):
    geom_names = sorted(df["geometry"].unique())
    load_cases = sorted(df["load_case"].unique())
    geom_map = {name: idx for idx, name in enumerate(geom_names)}
    load_map = {name: idx for idx, name in enumerate(load_cases)}

    geom_one_hot = np.eye(len(geom_names))[df["geometry"].map(geom_map)]
    load_one_hot = np.eye(len(load_cases))[df["load_case"].map(load_map)]

    screw_data = df[SCREW_COLS].values.astype(np.float32)
    X = np.hstack([screw_data, geom_one_hot, load_one_hot]).astype(np.float32)
    y = df["compliance"].values.astype(np.float32).reshape(-1, 1)
    meta = {
        "geom_names": geom_names,
        "load_cases": load_cases,
        "input_dim": X.shape[1],
    }
    return X, y, meta


class SurrogateModel(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_model(X: np.ndarray, y: np.ndarray, hidden: int, epochs: int, lr: float):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)

    X_std[X_std == 0] = 1.0
    y_std[y_std == 0] = 1.0

    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std

    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32)

    model = SurrogateModel(X_tensor.shape[1], hidden)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = loss_fn(pred, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % max(epochs // 10, 1) == 0:
            print(f"Epoch {epoch}/{epochs} -> loss={loss.item():.6f}")

    stats = {
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return model, stats


def save_artifacts(model, stats: dict, meta: dict):
    torch.save(model.state_dict(), MODEL_PATH)
    np.savez(STATS_PATH, **stats)
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"✅ Saved model to {MODEL_PATH}")
    print(f"✅ Saved stats to {STATS_PATH}")
    print(f"✅ Saved metadata to {META_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Train a multi-geometry PINN.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {args.data_dir}")

    df = load_dataset(files)
    X, y, meta = encode_features(df)
    meta["num_samples"] = int(len(df))
    meta["hidden_size"] = args.hidden_size

    print(f"Loaded {meta['num_samples']} samples "
          f"({len(meta['geom_names'])} geometries / {len(meta['load_cases'])} load cases).")

    model, stats = train_model(X, y, args.hidden_size, args.epochs, args.lr)
    save_artifacts(model, stats, meta)


if __name__ == "__main__":
    main()

