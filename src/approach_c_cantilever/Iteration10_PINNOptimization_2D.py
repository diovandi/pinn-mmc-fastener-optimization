import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import subprocess
import sys

# Ensure we can reuse SurrogateModel definition
sys.path.insert(0, str(Path(__file__).parent))
from TrainPINN_Cantilever_2D import SurrogateModel, HIDDEN_SIZES

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR = ROOT_DIR / "data/cantilever"
MODEL_PATH = ARTIFACT_DIR / "pinn_cantilever_2d.pth"
STATS_PATH = ARTIFACT_DIR / "norm_stats_cantilever_2d.npz"
JULIA_SRC = BASE_DIR / "CantileverDiffFEA.jl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load normalization stats and model weights
stats = np.load(STATS_PATH)
X_mean = stats["X_mean"]
X_std = stats["X_std"]
y_mean = stats["y_mean"]
y_std = stats["y_std"]

model = SurrogateModel(HIDDEN_SIZES)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


def evaluate_fea_compliance(position: float) -> float:
    """Call Julia solver to validate compliance at the given support position."""
    cmd = [
        "julia",
        "-e",
        (
            f'include("{JULIA_SRC}"); '
            f'println(solve_beam_compliance_2d([{position}]));'
        ),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Take the last line and parse as float
    last_line = result.stdout.strip().splitlines()[-1]
    return float(last_line)


def optimize_with_pinn(initial_pos=0.30, lr=5e-3, max_iter=25, tol=1e-6):
    pos = initial_pos
    records = []

    print("=" * 60)
    print("Iteration 10 (2D): PINN-Based Optimization")
    print("=" * 60)
    print(f"\nInitial support position: {pos:.4f} m\n")

    for it in range(max_iter):
        pos_norm = (np.array([[pos]], dtype=np.float32) - X_mean) / X_std

        pos_tensor = torch.tensor(pos_norm, dtype=torch.float32, requires_grad=True)
        c_norm = model(pos_tensor)
        c_val = (c_norm * y_std[0] + y_mean[0]).item()

        grad_norm = torch.autograd.grad(c_norm, pos_tensor, create_graph=False)[0].item()
        grad = grad_norm * y_std[0] / X_std[0]

        records.append((it, pos, c_val, grad))

        if it <= 5 or it % 5 == 0:
            print(f"Iter {it:2d}: pos={pos:.4f} m | C={c_val:.6e} J | grad={grad:.6e}")

        if abs(grad) < tol:
            print(f"\n✅ Converged at iteration {it} (|grad| < tol)")
            break

        pos -= lr * grad
        pos = float(np.clip(pos, 0.15, 0.85))

    fea_validation = evaluate_fea_compliance(pos)
    print(f"\nPINN-optimal position: {pos:.4f} m")
    print(f"PINN-predicted compliance: {records[-1][2]:.6e} J")
    print(f"2D FEA compliance at PINN optimum: {fea_validation:.6e} J")

    return records, pos, fea_validation


def write_output(records, final_pos, fea_compliance):
    output_file = OUTPUT_DIR / "iter_10_pinn_optimization_2d.txt"
    with open(output_file, "w") as f:
        f.write("Iteration 10 (2D): PINN-Based Optimization\n")
        f.write("-" * 60 + "\n")
        f.write(f"Initial support position: {records[0][1]:.4f}\n\n")
        for it, pos, comp, grad in records:
            f.write(f"PINN Iter {it:2d}: pos={pos:.4f}, C={comp:.6e}, grad={grad:.6e}\n")
        f.write("\n")
        f.write(f"Converged support position (PINN): {final_pos:.6f} m\n")
        f.write(f"PINN compliance: {records[-1][2]:.6e} J\n")
        f.write(f"2D FEA compliance at this position: {fea_compliance:.6e} J\n")

    print(f"\n✅ Saved PINN optimization log -> {output_file}")


if __name__ == "__main__":
    history, final_pos, fea_c = optimize_with_pinn()
    write_output(history, final_pos, fea_c)


