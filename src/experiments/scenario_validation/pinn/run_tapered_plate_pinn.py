"""
Inference harness that reuses the trained PINN to evaluate the tapered cantilever plate
geometry without modifying the original training code.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_ROOT = PROJECT_ROOT / "scenario_validation"
ARTIFACT_DIR = PROJECT_ROOT / "approach_a_pinn" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "pinn_model.pth"
STATS_PATH = ARTIFACT_DIR / "norm_stats.npz"
RESULTS_DIR = SCENARIO_ROOT / "results" / "pinn_new_part"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Geometry constants (millimeters / Newtons)
PLATE_LENGTH = 140.0
PLATE_HEIGHT = 60.0
ROW_A_X = 20.0
ROW_B_X = 80.0
ROW_SPACING = 15.0
ROW_CLEARANCE = 12.0
CLAMP_REGION = 10.0
TIP_LOAD_N = 450.0
TORSION_NM = 25.0


@dataclass
class ScrewLayout:
    s1: Tuple[float, float]
    s2: Tuple[float, float]

    def as_array(self) -> np.ndarray:
        return np.array([self.s1[0], self.s1[1], self.s2[0], self.s2[1]], dtype=np.float32)


class SurrogateModel(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_model() -> Tuple[SurrogateModel, dict]:
    if not MODEL_PATH.exists() or not STATS_PATH.exists():
        raise FileNotFoundError("PINN weights or normalization stats are missing. Train first.")

    model = SurrogateModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    stats = np.load(STATS_PATH)
    normalization = {
        "X_mean": torch.tensor(stats["X_mean"], dtype=torch.float32),
        "X_std": torch.tensor(stats["X_std"], dtype=torch.float32),
        "y_mean": torch.tensor(stats["y_mean"], dtype=torch.float32),
        "y_std": torch.tensor(stats["y_std"], dtype=torch.float32),
    }
    return model, normalization


def enumerate_row_positions(x_coord: float) -> List[Tuple[float, float]]:
    """Generate equally spaced y-coordinates for a screw row."""
    usable_height = PLATE_HEIGHT - 2 * ROW_CLEARANCE
    count = int(usable_height // ROW_SPACING) + 1
    start_y = ROW_CLEARANCE
    positions = []
    for i in range(count):
        y_pos = start_y + i * ROW_SPACING
        positions.append((x_coord, y_pos))
    return positions


def sample_layouts(num_samples: int) -> List[ScrewLayout]:
    row_a = enumerate_row_positions(ROW_A_X)
    row_b = enumerate_row_positions(ROW_B_X)
    layouts: List[ScrewLayout] = []
    for _ in range(num_samples):
        s1 = random.choice(row_a)
        s2 = random.choice(row_b)
        layouts.append(ScrewLayout(s1=s1, s2=s2))
    return layouts


def predict_compliance(
    model: SurrogateModel,
    normalization: dict,
    layouts: List[ScrewLayout],
) -> pd.DataFrame:
    X = np.stack([layout.as_array() for layout in layouts])
    X_norm = (X - normalization["X_mean"].numpy()) / normalization["X_std"].numpy()
    with torch.no_grad():
        y_pred_norm = model(torch.tensor(X_norm, dtype=torch.float32)).numpy().flatten()
    y_pred = y_pred_norm * normalization["y_std"].item() + normalization["y_mean"].item()
    df = pd.DataFrame(
        {
            "s1_x": X[:, 0],
            "s1_y": X[:, 1],
            "s2_x": X[:, 2],
            "s2_y": X[:, 3],
            "pinn_predicted_compliance": y_pred,
            "load_tip_force_N": TIP_LOAD_N,
            "load_torsion_Nm": TORSION_NM,
        }
    )
    return df


def export_geometry_manifest():
    manifest = {
        "geometry": "tapered_cantilever_plate",
        "dimensions_mm": {"length": PLATE_LENGTH, "height": PLATE_HEIGHT, "thickness_range": [6.0, 3.0]},
        "rows": {
            "row_a": {"x_mm": ROW_A_X, "spacing_mm": ROW_SPACING},
            "row_b": {"x_mm": ROW_B_X, "spacing_mm": ROW_SPACING},
        },
        "loads": {"tip_force_N": TIP_LOAD_N, "torsion_Nm": TORSION_NM},
        "clamp_region_mm": CLAMP_REGION,
    }
    manifest_path = SCENARIO_ROOT / "results" / "pinn_new_part" / "tapered_plate_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def run(num_samples: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    export_geometry_manifest()
    model, normalization = load_model()
    layouts = sample_layouts(num_samples)
    results = predict_compliance(model, normalization, layouts)
    output_file = RESULTS_DIR / "pinn_tapered_plate_predictions.csv"
    results.to_csv(output_file, index=False)
    print(f"✅ Saved {len(results)} PINN predictions to {output_file}")
    print(
        f"   Compliance range: {results['pinn_predicted_compliance'].min():.2f} – "
        f"{results['pinn_predicted_compliance'].max():.2f} J"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained PINN on tapered plate scenario.")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of screw layouts to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    run(num_samples=args.num_samples, seed=args.seed)

