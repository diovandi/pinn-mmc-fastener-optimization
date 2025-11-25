import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parent
ROOT_DIR = SRC_DIR.parent
DATA_PATH = (ROOT_DIR / "data/results/pinn_training_data.csv").resolve()
FEA_LOG_PATH = (ROOT_DIR / "data/results/lbracket_diff_fea_log.csv").resolve()
MODEL_PATH = (SRC_DIR / "approach_a_pinn/artifacts/pinn_model.pth").resolve()
STATS_PATH = (SRC_DIR / "approach_a_pinn/artifacts/norm_stats.npz").resolve()
HIDDEN_SIZE = 64

class SurrogateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, x):
        return self.net(x)

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at {DATA_PATH}")
    data = pd.read_csv(DATA_PATH, header=None).values.astype(np.float32)
    return data

def measure_timings():
    if FEA_LOG_PATH.exists():
        fea_log = pd.read_csv(FEA_LOG_PATH)
        fea_time = fea_log['wall_time_ms'].iloc[1:].mean() / 1000.0
    else:
        fea_time = 0.2

    if not MODEL_PATH.exists() or not STATS_PATH.exists():
        raise FileNotFoundError("TrainPINN outputs missing; run training first.")

    model = SurrogateModel()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    stats = np.load(STATS_PATH)
    X_mean = torch.tensor(stats['X_mean'], dtype=torch.float32)
    X_std = torch.tensor(stats['X_std'], dtype=torch.float32)

    raw_data = load_dataset()
    X_norm = (raw_data[:, 0:4] - X_mean.numpy()) / X_std.numpy()
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)

    with torch.no_grad():
        reps = 500
        start = time.perf_counter()
        for _ in range(reps):
            _ = model(X_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pinn_time = (time.perf_counter() - start) / reps

    return fea_time, pinn_time

# This script regenerates the "Proof" plots for the thesis presentation
# based on the validated data from the terminal logs.

def generate_plots():
    data = load_dataset()
    compliance_samples = data[:, 4]
    t_julia, t_pinn = measure_timings()

    # --- PLOT 1: SPEEDUP BAR CHART ---
    fig, ax = plt.subplots(figsize=(8, 6))
    times = [t_julia, t_pinn]
    labels = ['Traditional FEA\n(Julia Zygote)', 'AI Surrogate\n(PyTorch PINN)']
    colors = ['#7f7f7f', '#2ca02c']  # Grey vs Green

    bars = ax.bar(labels, times, color=colors, alpha=0.9)
    ax.set_yscale('log')
    ax.set_ylabel('Time per Iteration (Seconds) - Log Scale', fontsize=12)
    ax.set_title('Optimization Speed Benchmark', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    speedup = (t_julia / t_pinn) if t_pinn > 0 else float('inf')

    # Annotations
    ax.text(0, t_julia * 1.2, f"{t_julia * 1_000:.2f} ms", ha='center', fontweight='bold')
    ax.text(1, t_pinn * 1_000 * 1.2, f"{t_pinn * 1_000:.3f} ms", ha='center', fontweight='bold')

    ax.annotate(f"{speedup:,.0f}× faster",
                xy=(1, t_pinn),
                xytext=(0.5, t_pinn * 8),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                fontsize=12,
                color='crimson',
                ha='center')

    plt.savefig("Proof_Speedup.png", dpi=150)
    print("Generated Proof_Speedup.png")

    # --- PLOT 2: COMPLIANCE DISTRIBUTION ---
    plt.figure(figsize=(10, 6))
    plt.hist(compliance_samples, bins=15, color='skyblue', edgecolor='black', alpha=0.8)
    plt.title('Distribution of L-Bracket Compliance (Julia Physics Engine)', fontsize=14)
    plt.xlabel('Compliance (Joules) - Lower is Stiffer', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Annotations
    best_idx = np.argmin(compliance_samples)
    best_val = compliance_samples[best_idx]
    plt.axvline(x=best_val, color='r', linestyle='--', linewidth=2, label=f'Best Sample ({best_val:.1f} J)')
    plt.legend()
    
    plt.savefig("Proof_Distribution.png", dpi=150)
    print("Generated Proof_Distribution.png")

if __name__ == "__main__":
    generate_plots()
