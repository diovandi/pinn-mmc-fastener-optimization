import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "cantilever"
FIG_DIR = ROOT_DIR / "figures" / "cantilever"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def draw_cantilever_setup():
    """
    Draw a schematic of the 2D cantilever beam test case,
    similar in style to L_Bracket_Setup_Visual.png.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Geometry (in mm)
    L = 1000.0  # length
    h = 50.0    # height

    # Beam rectangle (0,0) to (L,h)
    beam = patches.Rectangle((0, 0), L, h,
                             facecolor='#d9d9d9', edgecolor='black',
                             linewidth=2, label='Steel Beam')
    ax.add_patch(beam)

    # Fixed support at left edge (x=0)
    support_x = 0.0
    ax.plot([support_x, support_x], [0, h], color='red', linewidth=4,
            label='Fixed Support (u=0)')
    # Hash marks for fixed support
    for y in np.linspace(0, h, 8):
        ax.plot([support_x, support_x - 10], [y, y + 5],
                color='red', linewidth=1)

    # Distributed load q along the top edge (downward)
    q_text = "q = 1000 N/m"
    top_y = h
    arrow_length = 15.0
    x_positions = np.linspace(100, L - 100, 8)
    for x in x_positions:
        ax.arrow(x, top_y + arrow_length, 0, -arrow_length,
                 head_width=15, head_length=5,
                 fc='blue', ec='blue', linewidth=1.5,
                 length_includes_head=True)
    ax.text(L / 2, top_y + arrow_length + 10, q_text,
            color='blue', fontsize=11, ha='center', fontweight='bold')

    # Example intermediate support at midspan (for visualization)
    support_x_mid = L / 2
    ax.plot([support_x_mid, support_x_mid], [0, -10],
            color='green', linewidth=3, label='Intermediate Support (test case)')
    ax.plot(support_x_mid, 0, marker='^', color='green', markersize=10)

    # Dimensions
    ax.annotate("", xy=(0, -25), xytext=(L, -25),
                arrowprops=dict(arrowstyle="<->"))
    ax.text(L / 2, -35, "1.0 m (1000 mm)", ha='center')

    ax.annotate("", xy=(-25, 0), xytext=(-25, h),
                arrowprops=dict(arrowstyle="<->"))
    ax.text(-35, h / 2, "0.05 m\n(50 mm)", va='center', ha='center', rotation=90)

    # Formatting
    ax.set_xlim(-80, L + 100)
    ax.set_ylim(-60, h + 80)
    ax.set_aspect('equal')
    ax.set_title("Cantilever Beam Test Case\n(2D Plane Stress)", fontsize=14, fontweight='bold')
    ax.set_xlabel("X Position (mm)")
    ax.set_ylabel("Y Position (mm)")
    ax.grid(True, linestyle=":", alpha=0.6)

    from matplotlib.lines import Line2D
    custom_lines = [
        patches.Patch(facecolor='#d9d9d9', edgecolor='black'),
        Line2D([0], [0], color='red', lw=4),
        Line2D([0], [0], color='blue', lw=2, marker='v'),
        Line2D([0], [0], color='green', lw=2, marker='^'),
    ]
    ax.legend(custom_lines,
              ['Steel Beam', 'Fixed Support', 'Distributed Load', 'Intermediate Support'],
              loc='upper right', fontsize=9)

    out_path = FIG_DIR / "Cantilever_Setup_Visual.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Diagram generated: {out_path}")


def draw_cantilever_training_overview():
    """
    Visualise training data and optimized support positions
    along the actual cantilever beam geometry.
    """
    # Load training data (1D and 2D if available)
    path_1d = DATA_DIR / "iter_6_training_data.csv"
    path_2d = DATA_DIR / "iter_6_training_data_2d.csv"
    df1 = pd.read_csv(path_1d) if path_1d.exists() else None
    df2 = pd.read_csv(path_2d) if path_2d.exists() else None

    # Load final optimized positions from Iteration 2-4 (1D & 2D) if available
    path_opt_1d = DATA_DIR / "iter_2-4_optimization.csv"
    path_opt_2d = DATA_DIR / "iter_2-4_optimization_2d.csv"
    df_opt_1d = pd.read_csv(path_opt_1d) if path_opt_1d.exists() else None
    df_opt_2d = pd.read_csv(path_opt_2d) if path_opt_2d.exists() else None
    
    # Load PINN optima
    import re
    pinn_1d_pos = None
    pinn_2d_pos = None
    path_pinn_1d = DATA_DIR / "iter_10_pinn_optimization.txt"
    path_pinn_2d = DATA_DIR / "iter_10_pinn_optimization_2d.txt"
    if path_pinn_1d.exists():
        with open(path_pinn_1d, 'r') as f:
            for line in f:
                m = re.search(r'Converged Position.*?(\d+\.\d+)', line)
                if m:
                    pinn_1d_pos = float(m.group(1))
                    break
    if path_pinn_2d.exists():
        with open(path_pinn_2d, 'r') as f:
            for line in f:
                m = re.search(r'PINN-optimal position.*?(\d+\.\d+)', line)
                if m:
                    pinn_2d_pos = float(m.group(1))
                    break

    fig, ax = plt.subplots(figsize=(10, 3))

    L = 1000.0
    h = 50.0

    # Draw beam
    beam = patches.Rectangle((0, 0), L, h,
                             facecolor='#f2f2f2', edgecolor='black',
                             linewidth=2)
    ax.add_patch(beam)

    # Training samples as dots along mid-height
    y_mid = h / 2.0
    if df1 is not None:
        ax.scatter(df1["support_pos"] * 1000.0,
                   np.full(len(df1), y_mid + 5),
                   c="tab:blue", s=25, alpha=0.7, edgecolors="black",
                   label="1D training samples")
    if df2 is not None:
        ax.scatter(df2["support_pos"] * 1000.0,
                   np.full(len(df2), y_mid - 5),
                   c="tab:orange", s=25, alpha=0.7, edgecolors="black",
                   marker="s", label="2D training samples")

    # Final optimized support positions (1D & 2D FEA)
    if df_opt_1d is not None:
        pos_col = "support_pos" if "support_pos" in df_opt_1d.columns else "position"
        x_opt_1d = float(df_opt_1d[pos_col].iloc[-1]) * 1000.0
        ax.plot(x_opt_1d, 0, marker='^', color='green', markersize=12,
                markeredgecolor='black', markeredgewidth=1.5,
                label=f"1D FEA optimum ({x_opt_1d/1000:.3f} m)", zorder=10)
        ax.text(x_opt_1d, -5, f"{x_opt_1d/1000:.3f}m", ha='center', fontsize=8, color='green', weight='bold')
    
    if df_opt_2d is not None:
        pos_col = "support_pos" if "support_pos" in df_opt_2d.columns else "position"
        x_opt_2d = float(df_opt_2d[pos_col].iloc[-1]) * 1000.0
        ax.plot(x_opt_2d, h, marker='v', color='purple', markersize=12,
                markeredgecolor='black', markeredgewidth=1.5,
                label=f"2D FEA optimum ({x_opt_2d/1000:.3f} m)", zorder=10)
        ax.text(x_opt_2d, h + 8, f"{x_opt_2d/1000:.3f}m", ha='center', fontsize=8, color='purple', weight='bold')
    
    # PINN optima
    if pinn_1d_pos is not None:
        ax.plot(pinn_1d_pos * 1000.0, 0, marker='*', color='darkgreen', markersize=10,
                markeredgecolor='black', markeredgewidth=1,
                label=f"1D PINN ({pinn_1d_pos:.3f} m)", zorder=9)
    
    if pinn_2d_pos is not None:
        ax.plot(pinn_2d_pos * 1000.0, h, marker='*', color='darkviolet', markersize=10,
                markeredgecolor='black', markeredgewidth=1,
                label=f"2D PINN ({pinn_2d_pos:.3f} m)", zorder=9)

    # Formatting
    ax.set_xlim(-50, L + 50)
    ax.set_ylim(-15, h + 25)
    ax.set_aspect('equal')
    ax.set_xlabel("X Position (mm)")
    ax.set_yticks([])
    ax.set_title("Cantilever Beam: Training Samples & Optimization Optima",
                 fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right", ncol=1)

    out_path = FIG_DIR / "Cantilever_Training_Overview.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"✅ Diagram generated: {out_path}")


if __name__ == "__main__":
    draw_cantilever_setup()
    draw_cantilever_training_overview()


