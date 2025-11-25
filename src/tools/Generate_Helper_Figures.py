"""
Helper figure generator for thesis visuals.
Creates:
1. Screw layout overview for each method/load case.
2. MMC screw trajectory plots for both load cases.
3. Load case schematic (horizontal vs vertical).
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
RESULTS_DIR = ROOT_DIR / "data" / "results"
ANSYS_PAYLOADS = RESULTS_DIR / "ansys_payloads.json"
MMC_LBRACKET_LOG = RESULTS_DIR / "mmc_lbracket_log.csv"
MMC_VERTICAL_LOG = RESULTS_DIR / "mmc_log_vertical.csv"

# --- Shared drawing utilities -------------------------------------------------

def draw_bracket_outline(ax):
    """Draw the L-bracket silhouette on provided axes."""
    total = 100
    thick = 25
    vertices = [
        (0, 0), (total, 0), (total, thick),
        (thick, thick), (thick, total), (0, total)
    ]
    poly = patches.Polygon(
        vertices,
        closed=True,
        facecolor="#f0f0f0",
        edgecolor="black",
        linewidth=1.5
    )
    ax.add_patch(poly)
    ax.set_xlim(-5, total + 5)
    ax.set_ylim(-5, total + 5)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

def annotate_support(ax):
    """Mark the fixed boundary along the vertical leg."""
    support_x = [0, 25]
    y = 100
    ax.plot(support_x, [y, y], color="red", linewidth=3, label="Fixed edge")
    for i in range(0, 26, 5):
        ax.plot([i, i - 2], [y, y + 3], color="red", linewidth=1)

def annotate_void(ax):
    """Draw void region box."""
    void = patches.Rectangle(
        (25, 25), 75, 75, linewidth=1, linestyle="--",
        facecolor="none", edgecolor="gray"
    )
    ax.add_patch(void)

# --- Figure generators --------------------------------------------------------

def generate_screw_layout_overview():
    """Visualize optimal screw placements for each method."""
    if not ANSYS_PAYLOADS.exists():
        raise FileNotFoundError(f"Missing payloads file at {ANSYS_PAYLOADS}")
    data = json.loads(ANSYS_PAYLOADS.read_text())
    layouts = data.get("layouts", [])
    if not layouts:
        raise ValueError("No layouts found in payload file.")

    cols = len(layouts)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 5), sharex=True, sharey=True)
    if cols == 1:
        axes = [axes]

    for ax, layout in zip(axes, layouts):
        draw_bracket_outline(ax)
        annotate_support(ax)
        annotate_void(ax)

        screws = layout.get("screws", [])
        xs = [s["x"] for s in screws]
        ys = [s["y"] for s in screws]
        ax.scatter(xs, ys, color="#1f77b4", s=80, zorder=5)
        for idx, (x, y) in enumerate(zip(xs, ys), start=1):
            ax.text(x + 1, y + 1, f"S{idx}", fontsize=9, weight="bold")

        method = layout.get("method", "unknown").upper()
        tag = layout.get("tag", "")
        compliance = layout.get("compliance", None)
        subtitle = f"{method} ({tag})"
        if compliance is not None:
            subtitle += f"\nCompliance: {compliance:.3f}"
        ax.set_title(subtitle, fontsize=11)

    fig.suptitle("Optimal Screw Placements Across Methods", fontsize=14, weight="bold")
    fig.tight_layout()
    output = BASE_DIR / "Screw_Position_Overview.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"✅ Saved screw layout overview to {output}")

def plot_mmc_trajectory(ax, csv_path, title):
    """Plot MMC screw movement trajectories using log data."""
    df = pd.read_csv(csv_path)
    draw_bracket_outline(ax)
    annotate_support(ax)
    annotate_void(ax)

    for screw_label, color in zip(["Screw 1", "Screw 2"], ["#d62728", "#2ca02c"]):
        xs = df["x1"] if screw_label == "Screw 1" else df["x2"]
        ys = df["y1"] if screw_label == "Screw 1" else df["y2"]
        ax.plot(xs, ys, "-o", label=screw_label, color=color, markersize=3)
        ax.text(xs.iloc[0], ys.iloc[0], "Start", color=color, fontsize=8)
        ax.text(xs.iloc[-1], ys.iloc[-1], "Final", color=color, fontsize=8)

    ax.set_title(title)
    ax.legend(loc="lower right")

def generate_mmc_trajectory_figures():
    """Create MMC screw trajectory figure for horizontal + vertical load cases."""
    available_logs = [(MMC_LBRACKET_LOG, "Horizontal Tip Load")]
    if MMC_VERTICAL_LOG.exists():
        available_logs.append((MMC_VERTICAL_LOG, "Vertical Tip Load"))

    fig, axes = plt.subplots(1, len(available_logs), figsize=(6 * len(available_logs), 5), sharex=True, sharey=True)
    if len(available_logs) == 1:
        axes = [axes]

    for ax, (path, title) in zip(axes, available_logs):
        plot_mmc_trajectory(ax, path, f"MMC Screw Trajectory\n{title}")

    fig.suptitle("MMC Screw Trajectories Across Iterations", fontsize=14, weight="bold")
    fig.tight_layout()
    output = BASE_DIR / "MMC_Screw_Trajectories.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"✅ Saved MMC trajectories to {output}")

def generate_load_case_visual():
    """Show both horizontal and vertical load schematics."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    load_cases = [
        ("Horizontal Tip Load", (100, 12.5), (-20, 0), "→", (100, 12.5)),
        ("Vertical Tip Load", (100, 12.5), (0, -25), "↓", (100, 20))
    ]

    for ax, (title, origin, delta, arrow_label, text_pos) in zip(axes, load_cases):
        draw_bracket_outline(ax)
        annotate_support(ax)
        annotate_void(ax)
        ax.arrow(
            origin[0], origin[1],
            delta[0], delta[1],
            head_width=4, head_length=6,
            fc="navy", ec="navy", linewidth=2
        )
        ax.text(
            text_pos[0] + 2, text_pos[1] + 5,
            arrow_label + " Applied Load (1000 N)",
            color="navy", fontsize=10
        )
        ax.set_title(title)

    fig.suptitle("Corner Bracket Load Cases", fontsize=14, weight="bold")
    fig.tight_layout()
    output = BASE_DIR / "L_Bracket_Loading_Cases.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"✅ Saved load case schematic to {output}")

def main():
    generate_screw_layout_overview()
    generate_mmc_trajectory_figures()
    generate_load_case_visual()
    print("🎯 Helper figure generation complete.")

if __name__ == "__main__":
    main()

