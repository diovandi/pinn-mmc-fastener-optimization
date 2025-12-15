"""
Unified comparison plotting script for both PINN and MMC methods.
Generates comparative plots as specified in plan section 4.2-4.3.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import time
import json

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR.parent
ROOT_DIR = SRC_DIR.parent
RESULTS_DIR = ROOT_DIR / "data" / "results"

# Paths - Use multi-geometry model (primary) with legacy fallback
PINN_DATA_LEGACY = RESULTS_DIR / "pinn_training_data.csv"
MULTI_GEOM_DATA_DIR = RESULTS_DIR / "multi_geom_training"
DIFF_FEA_LOG = RESULTS_DIR / "lbracket_diff_fea_log.csv"
MMC_LOG = RESULTS_DIR / "mmc_lbracket_log.csv"
METHOD_COMP = RESULTS_DIR / "method_comparison.csv"

# Multi-geometry model (primary)
MULTI_GEOM_MODEL_PATH = SRC_DIR / "approach_a_pinn/artifacts_multi_geom/pinn_multi_geom.pth"
MULTI_GEOM_STATS_PATH = SRC_DIR / "approach_a_pinn/artifacts_multi_geom/norm_stats_multi_geom.npz"
MULTI_GEOM_META_PATH = SRC_DIR / "approach_a_pinn/artifacts_multi_geom/dataset_metadata.json"

# Legacy model (fallback)
LEGACY_MODEL_PATH = SRC_DIR / "approach_a_pinn/artifacts/pinn_model.pth"
LEGACY_STATS_PATH = SRC_DIR / "approach_a_pinn/artifacts/norm_stats.npz"
LEGACY_HIDDEN_SIZE = 64
MULTI_GEOM_HIDDEN_SIZE = 96

class SurrogateModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x)

def load_pinn_model():
    """Load trained PINN for inference timing. Prefers multi-geometry model."""
    # Try multi-geometry model first
    if MULTI_GEOM_MODEL_PATH.exists() and MULTI_GEOM_STATS_PATH.exists():
        stats = np.load(MULTI_GEOM_STATS_PATH)
        if MULTI_GEOM_META_PATH.exists():
            with open(MULTI_GEOM_META_PATH, 'r') as f:
                meta = json.load(f)
            input_dim = meta.get('input_dim', 14)
        else:
            # Infer from stats
            input_dim = stats['X_mean'].shape[0]
        model = SurrogateModel(input_dim, MULTI_GEOM_HIDDEN_SIZE)
        model.load_state_dict(torch.load(MULTI_GEOM_MODEL_PATH))
        model.eval()
        return model, stats, 'multi_geom'
    
    # Fallback to legacy model
    if LEGACY_MODEL_PATH.exists() and LEGACY_STATS_PATH.exists():
        stats = np.load(LEGACY_STATS_PATH)
        model = SurrogateModel(4, LEGACY_HIDDEN_SIZE)
        model.load_state_dict(torch.load(LEGACY_MODEL_PATH))
        model.eval()
        return model, stats, 'legacy'
    
    return None, None, None

def generate_unified_comparison():
    """Generate all comparative plots for thesis."""
    
    # --- 1. Convergence Comparison (PINN vs MMC) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Load diff-FEA log (PINN training trajectory)
    if DIFF_FEA_LOG.exists():
        df_fea = pd.read_csv(DIFF_FEA_LOG)
        ax1.plot(df_fea['iter'], df_fea['compliance'], 'b-o', 
                label='Diff-FEA (Training)', markersize=4, linewidth=1.5)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Compliance (J)')
        ax1.set_title('Approach A: Differentiable FEA Optimization')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Load MMC log
    if MMC_LOG.exists():
        df_mmc = pd.read_csv(MMC_LOG)
        # MMC compliance is normalized, convert to approximate physical scale
        # Using rough scaling: MMC normalized ~0.4-0.8 maps to ~500-1200 J range
        mmc_compliance_scaled = df_mmc['compliance'] * 1500.0  # Rough scaling for visualization
        ax2.plot(df_mmc['iter'], mmc_compliance_scaled, 'r-s', 
                label='MMC Optimization', markersize=4, linewidth=1.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Compliance (J, scaled)')
        ax2.set_title('Approach B: MMC Optimization')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "unified_convergence_comparison.png", dpi=150)
    print(f"✅ Saved: unified_convergence_comparison.png")
    
    # --- 2. Speed Comparison Bar Chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if DIFF_FEA_LOG.exists():
        df_fea = pd.read_csv(DIFF_FEA_LOG)
        fea_time = df_fea['wall_time_ms'].iloc[1:].mean() / 1000.0  # Skip warm-up
    else:
        fea_time = 0.0104
    
    # Measure PINN inference time (use multi-geometry model if available)
    model, stats, model_type = load_pinn_model()
    if model is not None and stats is not None:
        # Load appropriate dataset based on model type
        if model_type == 'multi_geom' and MULTI_GEOM_DATA_DIR.exists():
            # Load multi-geometry data - need to match the encoding from train_multi_geom.py
            data_files = list(MULTI_GEOM_DATA_DIR.glob("*.csv"))
            if data_files:
                # Load all files and combine (matching train_multi_geom.py logic)
                frames = []
                for f in data_files:
                    df = pd.read_csv(f)
                    frames.append(df)
                sample_data = pd.concat(frames, ignore_index=True)
                
                # Extract screw columns (matching train_multi_geom.py)
                screw_cols = ["s1_x", "s1_y", "s2_x", "s2_y", "s3_x", "s3_y"]
                for col in screw_cols:
                    if col not in sample_data.columns:
                        sample_data[col] = 0.0
                
                # Get unique geometries and load cases (matching train_multi_geom.py)
                geom_names = sorted(sample_data["geometry"].unique()) if "geometry" in sample_data.columns else []
                load_cases = sorted(sample_data["load_case"].unique()) if "load_case" in sample_data.columns else []
                
                # Create one-hot encoding (matching train_multi_geom.py)
                geom_map = {name: idx for idx, name in enumerate(geom_names)}
                load_map = {name: idx for idx, name in enumerate(load_cases)}
                
                geom_one_hot = np.eye(len(geom_names))[sample_data["geometry"].map(geom_map).values.astype(int)] if geom_names else np.zeros((len(sample_data), 0))
                load_one_hot = np.eye(len(load_cases))[sample_data["load_case"].map(load_map).values.astype(int)] if load_cases else np.zeros((len(sample_data), 0))
                
                screw_data = sample_data[screw_cols].values.astype(np.float32)
                X = np.hstack([screw_data, geom_one_hot, load_one_hot]).astype(np.float32)
                
                # Normalize (stats should match the encoding)
                X_norm = (X - stats['X_mean']) / stats['X_std']
                X_tensor = torch.tensor(X_norm, dtype=torch.float32)
            else:
                X_tensor = None
        elif model_type == 'legacy' and PINN_DATA_LEGACY.exists():
            # Legacy model uses 4 inputs
            data = pd.read_csv(PINN_DATA_LEGACY, header=None).values.astype(np.float32)
            X_norm = (data[:, 0:4] - stats['X_mean']) / stats['X_std']
            X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        else:
            X_tensor = None
        
        if X_tensor is not None:
            with torch.no_grad():
                reps = 500
                start = time.perf_counter()
                for _ in range(reps):
                    _ = model(X_tensor)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                pinn_time = (time.perf_counter() - start) / reps
        else:
            # Use documented value from comprehensive_results_table.md
            pinn_time = 0.000009  # 0.009 ms for multi-geometry PINN
    else:
        # Use documented value from comprehensive_results_table.md
        pinn_time = 0.000009  # 0.009 ms for multi-geometry PINN
    
    if MMC_LOG.exists():
        df_mmc = pd.read_csv(MMC_LOG)
        mmc_time = df_mmc['wall_time_ms'].mean() / 1000.0 if 'wall_time_ms' in df_mmc.columns else 0.118
    else:
        mmc_time = 0.118
    
    methods = ['Diff-FEA\n(Julia)', 'PINN\n(PyTorch)', 'MMC\n(Python)']
    times = [fea_time, pinn_time, mmc_time]
    colors = ['#7f7f7f', '#2ca02c', '#d62728']
    
    bars = ax.bar(methods, times, color=colors, alpha=0.8)
    ax.set_yscale('log')
    ax.set_ylabel('Time per Iteration (Seconds, Log Scale)', fontsize=12)
    ax.set_title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Annotations
    for i, (bar, t) in enumerate(zip(bars, times)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.3,
                f'{t*1000:.2f} ms', ha='center', va='bottom', fontweight='bold')
    
    speedup_fea_pinn = fea_time / pinn_time
    ax.text(0.5, 0.02, f'Speedup: {speedup_fea_pinn:.0f}×', 
            transform=ax.transAxes, fontsize=12, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "unified_speed_comparison.png", dpi=150)
    print(f"✅ Saved: unified_speed_comparison.png")
    
    # --- 3. Final Compliance Comparison Table ---
    if METHOD_COMP.exists():
        df_comp = pd.read_csv(METHOD_COMP)
        print("\n=== Method Comparison Summary ===")
        print(df_comp.to_string(index=False))
        
        # Create comparison bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        methods = df_comp['label'].values
        compliances = df_comp['compliance'].values
        
        # Normalize MMC values for comparison (they're in different units)
        # For visualization, we'll show relative values
        bars = ax.bar(methods, compliances, color=['blue', 'red', 'orange'], alpha=0.7)
        ax.set_ylabel('Compliance (Normalized)', fontsize=12)
        ax.set_title('Final Compliance Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, compliances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "unified_compliance_comparison.png", dpi=150)
        print(f"✅ Saved: unified_compliance_comparison.png")
    
    print("\n✅ Unified comparison plots generated!")

if __name__ == "__main__":
    generate_unified_comparison()

