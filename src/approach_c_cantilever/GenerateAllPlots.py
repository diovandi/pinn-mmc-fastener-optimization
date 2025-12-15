#!/usr/bin/env python3
"""
Generate visualization plots for all cantilever beam optimization iterations.
Creates comprehensive plots for each iteration step.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data" / "cantilever"
FIGURES_DIR = SCRIPT_DIR.parent.parent / "figures" / "cantilever"
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"

# Ensure directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PINN Model Loading
# ============================================================================

class SurrogateModel(nn.Module):
    """PINN surrogate model for cantilever beam."""
    def __init__(self, input_dim=1, hidden_sizes=[32, 64, 32], output_dim=1):
        super(SurrogateModel, self).__init__()
        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_dim))
        self.net = nn.Sequential(*layers)  # Use 'net' to match saved model
    
    def forward(self, x):
        return self.net(x)

def load_pinn_model():
    """Load trained PINN model and normalization stats."""
    model_path = ARTIFACTS_DIR / "pinn_cantilever.pth"
    stats_path = ARTIFACTS_DIR / "norm_stats_cantilever.npz"
    
    if not model_path.exists() or not stats_path.exists():
        return None, None
    
    # Load normalization stats
    stats = np.load(stats_path)
    X_mean = torch.tensor(stats['X_mean'], dtype=torch.float32)
    X_std = torch.tensor(stats['X_std'], dtype=torch.float32)
    y_mean = torch.tensor(stats['y_mean'], dtype=torch.float32)
    y_std = torch.tensor(stats['y_std'], dtype=torch.float32)
    
    # Load model
    model = SurrogateModel(input_dim=1, hidden_sizes=[32, 64, 32], output_dim=1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, (X_mean, X_std, y_mean, y_std)

def predict_pinn(support_pos, model, stats):
    """Predict compliance using PINN model."""
    if model is None or stats is None:
        return None
    
    X_mean, X_std, y_mean, y_std = stats
    
    # Normalize input
    x = torch.tensor([[support_pos]], dtype=torch.float32)
    x_norm = (x - X_mean) / X_std
    
    # Predict
    with torch.no_grad():
        y_norm = model(x_norm)
    
    # Denormalize
    y = y_norm * y_std + y_mean
    return y.item()

# ============================================================================
# Plot Generation Functions
# ============================================================================

def plot_iteration1_validation():
    """Plot Iteration 1: Gradient validation."""
    data_file = DATA_DIR / "iter_1_validation.txt"
    if not data_file.exists():
        print(f"⚠️  Skipping Iteration 1: {data_file} not found")
        return
    
    # Read text file and extract values
    with open(data_file, 'r') as f:
        content = f.read()
    
    # Extract gradient values (simplified parsing)
    import re
    zygote_match = re.search(r'Gradient \(Zygote\):\s*\[([\d.e-]+)\]', content) or re.search(r'Zygote gradient:\s*([\d.e-]+)', content)
    fd_match = re.search(r'Gradient \(Finite Diff\):\s*\[([\d.e-]+)\]', content) or re.search(r'Finite-difference gradient:\s*([\d.e-]+)', content)
    error_match = re.search(r'Relative Error:\s*([\d.e-]+)', content) or re.search(r'Relative error:\s*([\d.e-]+)', content)
    
    if not all([zygote_match, fd_match, error_match]):
        print(f"⚠️  Could not parse {data_file}")
        return
    
    zygote_grad = float(zygote_match.group(1))
    fd_grad = float(fd_match.group(1))
    error = float(error_match.group(1))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Zygote AD', 'Finite Difference']
    gradients = [zygote_grad, fd_grad]
    colors = ['#2ca02c', '#1f77b4']
    
    bars = ax.bar(methods, gradients, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Gradient Value', fontsize=12, fontweight='bold')
    ax.set_title('Iteration 1: Gradient Validation\nZygote vs Finite-Difference', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, val in zip(bars, gradients):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6e}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add error annotation
    ax.text(0.5, 0.95, f'Relative Error: {error:.2e} ({error*100:.4f}%)',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "iter_1_gradient_validation.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: iter_1_gradient_validation.png")
    plt.close()

def plot_iterations2_4_optimization():
    """Plot Iterations 2-4: Optimization convergence."""
    data_file = DATA_DIR / "iter_2-4_optimization.csv"
    if not data_file.exists():
        print(f"⚠️  Skipping Iterations 2-4: {data_file} not found")
        return
    
    df = pd.read_csv(data_file)
    
    # Check column names
    print(f"   Columns: {df.columns.tolist()}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Compliance convergence
    iter_col = 'iteration' if 'iteration' in df.columns else df.columns[0]
    comp_col = 'compliance' if 'compliance' in df.columns else df.columns[1]
    pos_col = 'support_pos' if 'support_pos' in df.columns else (df.columns[2] if len(df.columns) > 2 else None)
    
    ax1.plot(df[iter_col], df[comp_col], 'b-o', linewidth=2, markersize=6, label='Compliance')
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Compliance (J)', fontsize=12, fontweight='bold')
    ax1.set_title('Iterations 2-4: Compliance Convergence', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    
    # Plot 2: Support position evolution (if available)
    if pos_col:
        ax2.plot(df[iter_col], df[pos_col], 'r-s', linewidth=2, markersize=6, label='Support Position')
        ax2.axhline(y=0.5, color='g', linestyle='--', linewidth=2, label='Initial Position (0.5)')
        ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Support Position (m)', fontsize=12, fontweight='bold')
        ax2.set_title('Iterations 2-4: Support Position Evolution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=11)
    else:
        # If no position column, show compliance vs iteration again with different style
        ax2.plot(df[iter_col], df[comp_col], 'r-s', linewidth=2, markersize=6, label='Compliance')
        ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Compliance (J)', fontsize=12, fontweight='bold')
        ax2.set_title('Iterations 2-4: Compliance Convergence (Detail)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "iter_2-4_optimization.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: iter_2-4_optimization.png")
    plt.close()

def plot_iteration5_two_supports():
    """Plot Iteration 5: Two-support optimization."""
    data_file = DATA_DIR / "iter_5_two_supports.txt"
    if not data_file.exists():
        print(f"⚠️  Skipping Iteration 5: {data_file} not found")
        return
    
    # Read and parse text file
    with open(data_file, 'r') as f:
        content = f.read()
    
    import re
    final_match = re.search(r'Final support positions:\s*\[([\d.]+),\s*([\d.]+)\]', content)
    final_compliance_match = re.search(r'Final compliance:\s*([\d.e-]+)', content)
    
    if not all([final_match, final_compliance_match]):
        print(f"⚠️  Could not parse {data_file}")
        return
    
    pos1 = float(final_match.group(1))
    pos2 = float(final_match.group(2))
    compliance = float(final_compliance_match.group(1))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw beam
    beam_length = 1.0
    beam_height = 0.05
    beam_rect = patches.Rectangle((0, -beam_height/2), beam_length, beam_height,
                                  linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.5)
    ax.add_patch(beam_rect)
    
    # Plot supports
    ax.plot([pos1, pos2], [0, 0], 'rs', markersize=15, label=f'Supports (C={compliance:.4e} J)', zorder=5)
    ax.vlines([pos1, pos2], -beam_height/2 - 0.02, beam_height/2 + 0.02, 
              colors='red', linewidths=3, linestyles='solid', label='Support Positions')
    
    # Fixed end
    ax.plot([0, 0], [-beam_height/2 - 0.02, beam_height/2 + 0.02], 
            'k-', linewidth=4, label='Fixed End')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel('Position along beam (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Height (m)', fontsize=12, fontweight='bold')
    ax.set_title('Iteration 5: Two-Support Optimization\nFinal Configuration', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "iter_5_two_supports.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: iter_5_two_supports.png")
    plt.close()

def plot_iteration6_training_data():
    """Plot Iteration 6: Training data visualization."""
    data_file = DATA_DIR / "iter_6_training_data.csv"
    if not data_file.exists():
        print(f"⚠️  Skipping Iteration 6: {data_file} not found")
        return
    
    # Try reading with header first, then without
    try:
        df = pd.read_csv(data_file)
        if 'support_pos' in df.columns:
            pos_col = 'support_pos'
            comp_col = 'compliance'
        else:
            df = pd.read_csv(data_file, header=None, names=['support_position', 'compliance'])
            pos_col = 'support_position'
            comp_col = 'compliance'
    except:
        df = pd.read_csv(data_file, header=None, names=['support_position', 'compliance'])
        pos_col = 'support_position'
        comp_col = 'compliance'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Scatter plot
    scatter = ax1.scatter(df[pos_col], df[comp_col], 
                         c=df[comp_col], cmap='viridis', s=50, alpha=0.7, edgecolors='black')
    ax1.set_xlabel('Support Position (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Compliance (J)', fontsize=12, fontweight='bold')
    ax1.set_title('Iteration 6: Training Data Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    plt.colorbar(scatter, ax=ax1, label='Compliance (J)')
    
    # Plot 2: Histogram
    ax2.hist(df[pos_col], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Support Position (m)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Iteration 6: Support Position Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "iter_6_training_data.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: iter_6_training_data.png")
    plt.close()

def plot_iterations7_9_pinn_training():
    """Plot Iterations 7-9: PINN training curve."""
    data_file = DATA_DIR / "iter_7-9_pinn_training.csv"
    if not data_file.exists():
        print(f"⚠️  Skipping Iterations 7-9: {data_file} not found")
        return
    
    df = pd.read_csv(data_file)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['epoch'], df['loss'], 'b-', linewidth=2, label='Training Loss')
    if 'val_loss' in df.columns:
        ax.plot(df['epoch'], df['val_loss'], 'r--', linewidth=2, label='Validation Loss')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('Iterations 7-9: PINN Training Curve', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "iter_7-9_pinn_training.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: iter_7-9_pinn_training.png")
    plt.close()

def plot_iteration10_pinn_optimization():
    """Plot Iteration 10: PINN optimization."""
    data_file = DATA_DIR / "iter_10_pinn_optimization.txt"
    if not data_file.exists():
        print(f"⚠️  Skipping Iteration 10: {data_file} not found")
        return
    
    # Read and parse
    with open(data_file, 'r') as f:
        content = f.read()
    
    import re
    final_pos_match = re.search(r'PINN-optimized position:\s*([\d.]+)', content)
    final_compliance_match = re.search(r'PINN-optimized compliance:\s*([\d.e-]+)', content)
    
    if not all([final_pos_match, final_compliance_match]):
        print(f"⚠️  Could not parse {data_file}")
        return
    
    pinn_pos = float(final_pos_match.group(1))
    pinn_compliance = float(final_compliance_match.group(1))
    
    # Load PINN model to plot prediction curve
    model, stats = load_pinn_model()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot PINN prediction curve if available
    if model is not None and stats is not None:
        x_range = np.linspace(0.1, 0.9, 100)
        y_pred = [predict_pinn(x, model, stats) for x in x_range]
        ax.plot(x_range, y_pred, 'b-', linewidth=2, label='PINN Prediction', alpha=0.7)
    
    # Mark optimized point
    ax.plot(pinn_pos, pinn_compliance, 'ro', markersize=12, label=f'PINN Optimum (x={pinn_pos:.3f})', zorder=5)
    
    ax.set_xlabel('Support Position (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Compliance (J)', fontsize=12, fontweight='bold')
    ax.set_title('Iteration 10: PINN Optimization Result', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "iter_10_pinn_optimization.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: iter_10_pinn_optimization.png")
    plt.close()

def plot_iteration11_validation():
    """Plot Iteration 11: PINN vs FEA validation."""
    data_file = DATA_DIR / "iter_11_validation.txt"
    if not data_file.exists():
        print(f"⚠️  Skipping Iteration 11: {data_file} not found")
        return
    
    # Read and parse
    with open(data_file, 'r') as f:
        content = f.read()
    
    import re
    pinn_pos_match = re.search(r'PINN position:\s*([\d.]+)', content)
    fea_pos_match = re.search(r'FEA optimum position:\s*([\d.]+)', content)
    pinn_comp_match = re.search(r'PINN compliance:\s*([\d.e-]+)', content)
    fea_comp_match = re.search(r'FEA compliance:\s*([\d.e-]+)', content)
    
    if not all([pinn_pos_match, fea_pos_match, pinn_comp_match, fea_comp_match]):
        print(f"⚠️  Could not parse {data_file}")
        return
    
    pinn_pos = float(pinn_pos_match.group(1))
    fea_pos = float(fea_pos_match.group(1))
    pinn_comp = float(pinn_comp_match.group(1))
    fea_comp = float(fea_comp_match.group(1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Position comparison
    methods = ['PINN', 'FEA']
    positions = [pinn_pos, fea_pos]
    colors = ['#ff7f0e', '#2ca02c']
    
    bars1 = ax1.bar(methods, positions, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Support Position (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Iteration 11: Position Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars1, positions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 2: Compliance comparison
    compliances = [pinn_comp, fea_comp]
    bars2 = ax2.bar(methods, compliances, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Compliance (J)', fontsize=12, fontweight='bold')
    ax2.set_title('Iteration 11: Compliance Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, val in zip(bars2, compliances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4e}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "iter_11_validation.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: iter_11_validation.png")
    plt.close()

def plot_iteration12_multi_support():
    """Plot Iteration 12: C*(N) curve."""
    data_file = DATA_DIR / "iter_12_c_star_n_curve.csv"
    if not data_file.exists():
        print(f"⚠️  Skipping Iteration 12: {data_file} not found")
        return
    
    df = pd.read_csv(data_file)
    
    # Find column names
    n_col = 'n_supports' if 'n_supports' in df.columns else df.columns[0]
    comp_col = 'fea_optimum_compliance' if 'fea_optimum_compliance' in df.columns else ('optimal_compliance' if 'optimal_compliance' in df.columns else (df.columns[1] if len(df.columns) > 1 else None))
    
    if comp_col is None:
        print(f"⚠️  Could not find compliance column in {data_file}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df[n_col], df[comp_col], 'b-o', linewidth=2, markersize=10, label='Optimal Compliance')
    ax.set_xlabel('Number of Supports (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Optimal Compliance C* (J)', fontsize=12, fontweight='bold')
    ax.set_title('Iteration 12: C*(N) Curve\nDiminishing Returns with More Supports', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    ax.set_xticks(df[n_col])
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "iter_12_multi_support.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: iter_12_multi_support.png")
    plt.close()

def plot_iteration13_benchmark():
    """Plot Iteration 13: Computational benchmark."""
    data_file = DATA_DIR / "iter_13_benchmark.txt"
    if not data_file.exists():
        print(f"⚠️  Skipping Iteration 13: {data_file} not found")
        return
    
    # Read and parse
    with open(data_file, 'r') as f:
        content = f.read()
    
    import re
    fd_time_match = re.search(r'Finite-difference time:\s*([\d.e-]+)', content)
    zygote_time_match = re.search(r'Zygote time:\s*([\d.e-]+)', content)
    speedup_match = re.search(r'Speedup:\s*([\d.]+)', content)
    
    if not all([fd_time_match, zygote_time_match, speedup_match]):
        print(f"⚠️  Could not parse {data_file}")
        return
    
    fd_time = float(fd_time_match.group(1))
    zygote_time = float(zygote_time_match.group(1))
    speedup = float(speedup_match.group(1))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Finite-Difference', 'Zygote AD']
    times = [fd_time, zygote_time]
    colors = ['#d62728', '#2ca02c']
    
    bars = ax.bar(methods, times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Time per Gradient (s)', fontsize=12, fontweight='bold')
    ax.set_title(f'Iteration 13: Computational Benchmark\nSpeedup: {speedup:.1f}×', 
                 fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, linestyle='--', which='both')
    
    for bar, val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4e}s', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "iter_13_benchmark.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: iter_13_benchmark.png")
    plt.close()

def plot_iteration14_accuracy():
    """Plot Iteration 14: Accuracy comparison with PINN."""
    data_file = DATA_DIR / "iter_14_accuracy_comparison.txt"
    if not data_file.exists():
        print(f"⚠️  Skipping Iteration 14: {data_file} not found")
        return
    
    # Read and parse
    with open(data_file, 'r') as f:
        content = f.read()
    
    import re
    
    # Extract 1D results
    defl_1d_match = re.search(r'Julia 1D Beam Results:.*?Tip deflection:\s*([\d.e-]+)', content, re.DOTALL)
    comp_1d_match = re.search(r'Julia 1D Beam Results:.*?Compliance:\s*([\d.e-]+)', content, re.DOTALL)
    
    # Extract 2D results
    defl_2d_match = re.search(r'Julia 2D Continuum Results:.*?Tip deflection:\s*([\d.e-]+)', content, re.DOTALL)
    comp_2d_match = re.search(r'Julia 2D Continuum Results:.*?Compliance:\s*([\d.e-]+)', content, re.DOTALL)
    
    # Extract FreeFEM results
    defl_ff_match = re.search(r'FreeFEM 2D Results:.*?Tip deflection:\s*([\d.e-]+)', content, re.DOTALL)
    comp_ff_match = re.search(r'FreeFEM 2D Results:.*?Compliance:\s*([\d.e-]+)', content, re.DOTALL)
    
    # Extract PINN 2D results (if present)
    comp_pinn2d_match = re.search(r'PINN \(2D FEA\) Results:.*?Compliance:\s*([\d.e-]+)', content, re.DOTALL)
    
    if not all([defl_1d_match, comp_1d_match, defl_2d_match, comp_2d_match, defl_ff_match, comp_ff_match]):
        print(f"⚠️  Could not parse {data_file}")
        return
    
    defl_1d = float(defl_1d_match.group(1))
    comp_1d = float(comp_1d_match.group(1))
    defl_2d = float(defl_2d_match.group(1))
    comp_2d = float(comp_2d_match.group(1))
    defl_ff = float(defl_ff_match.group(1))
    comp_ff = float(comp_ff_match.group(1))
    comp_pinn2d = float(comp_pinn2d_match.group(1)) if comp_pinn2d_match else None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Tip deflection comparison
    methods = ['Julia 1D', 'Julia 2D', 'FreeFEM 2D']
    deflections = [defl_1d, defl_2d, defl_ff]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    bars1 = ax1.bar(methods, deflections, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Tip Deflection (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Iteration 14: Tip Deflection Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    
    for bar, val in zip(bars1, deflections):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}', ha='center', va='bottom', fontweight='bold', fontsize=9, rotation=90)
    
    # Plot 2: Compliance comparison (including PINN)
    methods_comp = ['Julia 1D', 'Julia 2D', 'FreeFEM 2D']
    compliances = [comp_1d, comp_2d, comp_ff]
    colors_comp = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Add PINN 2D bar if available
    if comp_pinn2d is not None:
        methods_comp.append('PINN 2D')
        compliances.append(comp_pinn2d)
        colors_comp.append('#d62728')
    
    bars2 = ax2.bar(methods_comp, compliances, color=colors_comp, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Compliance (J)', fontsize=12, fontweight='bold')
    ax2.set_title('Iteration 14: Compliance Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_yscale('log')
    
    for bar, val in zip(bars2, compliances):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}', ha='center', va='bottom', fontweight='bold', fontsize=9, rotation=90)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "iter_14_accuracy.png", dpi=150, bbox_inches='tight')
    print(f"✅ Saved: iter_14_accuracy.png")
    plt.close()

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Generate all plots."""
    print("=" * 70)
    print("Generating visualization plots for all cantilever iterations...")
    print("=" * 70)
    print()
    
    plot_iteration1_validation()
    plot_iterations2_4_optimization()
    plot_iteration5_two_supports()
    plot_iteration6_training_data()
    plot_iterations7_9_pinn_training()
    plot_iteration10_pinn_optimization()
    plot_iteration11_validation()
    plot_iteration12_multi_support()
    plot_iteration13_benchmark()
    plot_iteration14_accuracy()
    
    print()
    print("=" * 70)
    print("✅ All plots generated successfully!")
    print(f"   Figures saved to: {FIGURES_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()

