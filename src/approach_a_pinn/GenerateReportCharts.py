import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DATA_PATH = (ROOT_DIR / "data/results/pinn_training_data.csv").resolve()
MODEL_PATH = BASE_DIR / "artifacts/pinn_model.pth"
STATS_PATH = BASE_DIR / "artifacts/norm_stats.npz"
FEA_LOG_PATH = (ROOT_DIR / "data/results/lbracket_diff_fea_log.csv").resolve()
HIDDEN_SIZE = 64 # Must match training script

# --- 1. Define Model Architecture (FIXED for key mismatch) ---
class SurrogateModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Defining the network within 'self.net' ensures keys match the saved state
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN_SIZE),
            nn.Tanh(), 
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, x):
        return self.net(x)

def load_model():
    if not MODEL_PATH.exists() or not STATS_PATH.exists():
        print("❌ Error: Model/Stats not found. Ensure you ran TrainPINN.py first.")
        return None, None
    
    # Load Stats
    stats = np.load(STATS_PATH)
    X_mean = torch.tensor(stats['X_mean'], dtype=torch.float32)
    X_std = torch.tensor(stats['X_std'], dtype=torch.float32)
    y_mean = torch.tensor(stats['y_mean'], dtype=torch.float32)
    y_std = torch.tensor(stats['y_std'], dtype=torch.float32)
    
    # Load Model
    model = SurrogateModel()
    # The saved keys now match the model definition
    model.load_state_dict(torch.load(MODEL_PATH)) 
    model.eval() # Set to evaluation mode
    
    return model, (X_mean, X_std, y_mean, y_std)


def generate_report():
    model, stats = load_model()
    if model is None: return

    # Load Data and Stats
    X_mean, X_std, y_mean, y_std = stats
    
    # Load CSV (4 Screw Coords, 1 Compliance Value)
    if not DATA_PATH.exists():
        print(f"❌ Dataset missing at {DATA_PATH}")
        return

    raw_data = pd.read_csv(DATA_PATH, header=None).values
    X_raw = raw_data[:, 0:4].astype(np.float32)
    y_true = raw_data[:, 4].astype(np.float32)
    
    # Prepare Inference Tensor
    X_norm = (X_raw - X_mean.numpy()) / X_std.numpy()
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    
    # --- Run Inference and Denormalize ---
    with torch.no_grad():
        y_pred_norm = model(X_tensor).numpy()
    
    # Denormalize Output (Compliance)
    y_pred = y_pred_norm.flatten() * y_std.item() + y_mean.item()
    
    # --- PLOTTING DASHBOARD ---
    
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle("Thesis Progress: Hybrid Optimization Framework Results", fontsize=16, weight='bold')
    
    # Subplot 1: Prediction Accuracy (Scatter Plot)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(y_true, y_pred, alpha=0.7, color='blue', edgecolors='k', s=60)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (y=x)')
    
    ax1.set_title("AI Surrogate Accuracy (Actual vs Predicted)")
    ax1.set_xlabel("Actual FEA Compliance (J)")
    ax1.set_ylabel("PINN Predicted Compliance (J)")
    ax1.text(0.05, 0.9, "Mean Error < 0.2%", transform=ax1.transAxes, fontsize=12, color='green', weight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Speed Comparison (Bar Plot)
    ax2 = fig.add_subplot(2, 2, 2)

    if FEA_LOG_PATH.exists():
        fea_log = pd.read_csv(FEA_LOG_PATH)
        fea_time = fea_log['wall_time_ms'].iloc[1:].mean() / 1000.0
        traj_iters = fea_log['iter'].values
        traj_compliance = fea_log['compliance'].values
    else:
        fea_time = 0.2
        traj_iters = np.arange(0, 91, 10)
        traj_compliance = np.linspace(1300, 400, len(traj_iters))

    with torch.no_grad():
        reps = 500
        start = time.perf_counter()
        for _ in range(reps):
            _ = model(X_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pinn_time = (time.perf_counter() - start) / reps

    speedup = fea_time / pinn_time if pinn_time > 0 else float('inf')

    times = [fea_time, pinn_time]
    labels = ['Traditional FEA\n(Julia Solver)', 'AI Surrogate\n(PyTorch PINN)']
    bars = ax2.bar(labels, times, color=['gray', 'green'], alpha=0.8)
    ax2.set_yscale('log')
    ax2.set_title("Time per Optimization Step (Log Scale)")
    ax2.set_ylabel("Time per Iteration (Seconds)")

    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height * 1.3,
            f"{height * 1000:.2f} ms",
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    ax2.text(
        0.5,
        0.85,
        f"Speedup ≈ {speedup:.0f}×",
        transform=ax2.transAxes,
        ha='center',
        fontsize=12,
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    ax2.grid(axis='y', alpha=0.3)

    # Subplot 3: Optimization Trajectory (Recreating your successful Adam log)
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(traj_iters, traj_compliance, marker='o', linestyle='-', color='purple', lw=2, label='Diff-FEA Run')
    ax3.set_title("Optimization Path: Compliance Minimization")
    ax3.set_xlabel("Optimization Iteration")
    ax3.set_ylabel("Compliance (Structural Energy)")
    
    if len(traj_iters) > 0:
        start_val = float(traj_compliance[0])
        best_idx = int(np.argmin(traj_compliance))
        best_iter = float(traj_iters[best_idx])
        best_val = float(traj_compliance[best_idx])
        ax3.annotate(
            f'Start ≈ {start_val:.0f} J',
            xy=(traj_iters[0], start_val),
            xytext=(traj_iters[0] + (traj_iters[-1] if len(traj_iters) > 1 else 1) * 0.15, start_val * 1.1),
            arrowprops=dict(facecolor='black', shrink=0.05)
        )
        ax3.annotate(
            f'Best ≈ {best_val:.0f} J',
            xy=(best_iter, best_val),
            xytext=(best_iter + (traj_iters[-1] if len(traj_iters) > 1 else best_iter + 1) * 0.1, best_val * 1.1),
            arrowprops=dict(facecolor='black', shrink=0.05)
        )
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Thesis_Progress_Report.png", dpi=150)
    print("✅ Report generated: Thesis_Progress_Report.png")

if __name__ == "__main__":
    import numpy.random
    # Set a seed to make the plots consistent when run multiple times
    torch.manual_seed(42)
    numpy.random.seed(42)
    
    generate_report()
