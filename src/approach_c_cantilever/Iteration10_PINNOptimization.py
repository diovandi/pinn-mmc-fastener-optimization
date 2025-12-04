import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from TrainPINN_Cantilever import SurrogateModel, HIDDEN_SIZES

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR = ROOT_DIR / "data/cantilever"
MODEL_PATH = ARTIFACT_DIR / "pinn_cantilever.pth"
STATS_PATH = ARTIFACT_DIR / "norm_stats_cantilever.npz"

# Load model and stats
stats = np.load(STATS_PATH)
X_mean = stats['X_mean']
X_std = stats['X_std']
y_mean = stats['y_mean']
y_std = stats['y_std']

model = SurrogateModel(HIDDEN_SIZES)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Optimization using PINN
initial_pos = 0.3
lr = 0.01
max_iter = 20
tol = 1e-8

pos = initial_pos
results = []

print("=" * 50)
print("Iteration 10: PINN-Based Optimization")
print("=" * 50)
print(f"\nInitial support position: {initial_pos:.4f}\n")

for iter in range(max_iter):
    # Normalize input
    pos_norm = (np.array([[pos]]) - X_mean) / X_std
    pos_tensor = torch.tensor(pos_norm, dtype=torch.float32)
    
    # Forward pass
    with torch.no_grad():
        c_pred_norm = model(pos_tensor)
        c_pred = c_pred_norm.item() * y_std[0] + y_mean[0]
    
    # Compute gradient (differentiate through PINN)
    pos_tensor.requires_grad = True
    c_pred_norm = model(pos_tensor)
    c_pred_grad = c_pred_norm * y_std[0] + y_mean[0]
    grad = torch.autograd.grad(c_pred_grad, pos_tensor, create_graph=False)[0].item()
    grad = grad / X_std[0]  # Chain rule for normalization
    
    results.append((iter, pos, c_pred, grad))
    
    if iter <= 5 or iter % 5 == 0:
        print(f"PINN Iter {iter:2d}: pos={pos:.4f}, C={c_pred:.6e}, grad={grad:.6e}")
    
    # Check convergence
    if abs(grad) < tol:
        print(f"\nConverged at iteration {iter}")
        break
    
    # Gradient descent
    pos = pos - lr * grad
    pos = max(0.1, min(0.9, pos))  # Clamp to [0.1, 0.9]

print(f"\nConverged Position (PINN): {pos:.4f}")
print(f"PINN-predicted Compliance: {results[-1][2]:.6e}")

# Write output
output_file = OUTPUT_DIR / "iter_10_pinn_optimization.txt"
with open(output_file, 'w') as f:
    f.write("Iteration 10: PINN-Based Optimization\n")
    f.write("-" * 50 + "\n")
    f.write(f"Initial support position: {initial_pos:.4f}\n\n")
    for iter, pos_val, c, g in results:
        f.write(f"PINN Iter {iter:2d}: pos={pos_val:.4f}, C={c:.6e}\n")
    f.write(f"\nConverged Position (PINN): {pos:.4f}\n")
    f.write(f"PINN-predicted Compliance: {results[-1][2]:.6e}\n")

print(f"\n✅ Results saved to {output_file}")

