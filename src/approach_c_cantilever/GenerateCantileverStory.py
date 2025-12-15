#!/usr/bin/env python3
"""
Cantilever story figure: training data + optimization convergence + PINN training.

This script generates a multi-panel figure similar in spirit to
`figures/setup/Screw_Position_Overview.png`, but for the cantilever beam:

- Panel A: 1D vs 2D FEA training samples (support position vs compliance)
- Panel B: 1D diffFEA optimization convergence (compliance & support position vs iteration)
- Panel C: 1D & 2D PINN training curves (loss vs epoch)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "cantilever"
FIG_DIR = ROOT_DIR / "figures" / "cantilever"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_or_none(path: Path):
    return pd.read_csv(path) if path.exists() else None


def main():
    # --- Load data ---
    train_1d = load_or_none(DATA_DIR / "iter_6_training_data.csv")
    train_2d = load_or_none(DATA_DIR / "iter_6_training_data_2d.csv")
    opt_1d = load_or_none(DATA_DIR / "iter_2-4_optimization.csv")
    opt_2d = load_or_none(DATA_DIR / "iter_2-4_optimization_2d.csv")
    pinn_1d = load_or_none(DATA_DIR / "iter_7-9_pinn_training.csv")
    pinn_2d = load_or_none(DATA_DIR / "iter_7-9_pinn_training_2d.csv")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axA, axB, axC = axes

    # ============================================================
    # Panel A: Training data (1D vs 2D)
    # ============================================================
    if train_1d is not None or train_2d is not None:
        if train_1d is not None:
            axA.scatter(
                train_1d["support_pos"] * 1000.0,
                train_1d["compliance"],
                c="tab:blue",
                s=40,
                alpha=0.7,
                edgecolors="black",
                label="1D FEA samples",
            )
        if train_2d is not None:
            axA.scatter(
                train_2d["support_pos"] * 1000.0,
                train_2d["compliance"],
                c="tab:orange",
                s=40,
                alpha=0.7,
                edgecolors="black",
                marker="s",
                label="2D FEA samples",
            )

        axA.set_xlabel("Support position x (mm)", fontsize=11)
        axA.set_ylabel("Compliance C (J)", fontsize=11)
        axA.set_title("Training Data (1D vs 2D FEA)", fontsize=13, fontweight="bold")
        axA.grid(True, alpha=0.3, linestyle="--")
        axA.legend(fontsize=9)
    else:
        axA.text(0.5, 0.5, "No training data found", ha="center", va="center")
        axA.axis("off")

    # ============================================================
    # Panel B: Optimization convergence (1D & 2D)
    # ============================================================
    if opt_1d is not None or opt_2d is not None:
        axB2 = axB.twinx()
        
        # Plot 1D optimization
        if opt_1d is not None:
            cols = opt_1d.columns.tolist()
            iter_col = "iteration" if "iteration" in cols else cols[0]
            pos_col = (
                "support_pos"
                if "support_pos" in cols
                else ("position" if "position" in cols else cols[1])
            )
            comp_col = "compliance" if "compliance" in cols else cols[2]

            iters_1d = opt_1d[iter_col]
            pos_1d = opt_1d[pos_col]
            comp_1d = opt_1d[comp_col]

            axB.plot(
                iters_1d,
                comp_1d,
                "-o",
                color="tab:blue",
                label="1D Compliance",
                linewidth=2,
                markersize=4,
                alpha=0.8,
            )
            axB2.plot(
                iters_1d,
                pos_1d * 1000.0,
                "--s",
                color="tab:blue",
                label="1D Position",
                linewidth=1.5,
                markersize=4,
                alpha=0.6,
            )
        
        # Plot 2D optimization
        if opt_2d is not None:
            cols = opt_2d.columns.tolist()
            iter_col = "iteration" if "iteration" in cols else cols[0]
            pos_col = (
                "support_pos"
                if "support_pos" in cols
                else ("position" if "position" in cols else cols[1])
            )
            comp_col = "compliance" if "compliance" in cols else cols[2]

            iters_2d = opt_2d[iter_col]
            pos_2d = opt_2d[pos_col]
            comp_2d = opt_2d[comp_col]

            axB.plot(
                iters_2d,
                comp_2d,
                "-^",
                color="tab:orange",
                label="2D Compliance",
                linewidth=2,
                markersize=4,
                alpha=0.8,
            )
            axB2.plot(
                iters_2d,
                pos_2d * 1000.0,
                "--^",
                color="tab:orange",
                label="2D Position",
                linewidth=1.5,
                markersize=4,
                alpha=0.6,
            )

        axB.set_xlabel("Iteration", fontsize=11)
        axB.set_ylabel("Compliance C (J)", color="black", fontsize=11)
        axB.tick_params(axis="y", labelcolor="black")
        axB.grid(True, alpha=0.3, linestyle="--")

        axB2.set_ylabel("Support position x (mm)", color="black", fontsize=11)
        axB2.tick_params(axis="y", labelcolor="black")

        axB.set_title("Optimization Convergence (1D & 2D)", fontsize=13, fontweight="bold")

        # Combined legend
        lines1, labels1 = axB.get_legend_handles_labels()
        lines2, labels2 = axB2.get_legend_handles_labels()
        axB2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    else:
        axB.text(0.5, 0.5, "No optimization log found", ha="center", va="center")
        axB.axis("off")

    # ============================================================
    # Panel C: PINN training (1D & 2D)
    # ============================================================
    if pinn_1d is not None or pinn_2d is not None:
        if pinn_1d is not None:
            axC.plot(
                pinn_1d["epoch"],
                pinn_1d["loss"],
                "b-",
                linewidth=2,
                label="1D PINN - Train",
            )
            if "val_loss" in pinn_1d.columns:
                axC.plot(
                    pinn_1d["epoch"],
                    pinn_1d["val_loss"],
                    "b--",
                    linewidth=1.5,
                    label="1D PINN - Val",
                )

        if pinn_2d is not None:
            axC.plot(
                pinn_2d["epoch"],
                pinn_2d["loss"],
                "r-",
                linewidth=2,
                label="2D PINN - Train",
            )
            if "val_loss" in pinn_2d.columns:
                axC.plot(
                    pinn_2d["epoch"],
                    pinn_2d["val_loss"],
                    "r--",
                    linewidth=1.5,
                    label="2D PINN - Val",
                )

        axC.set_xlabel("Epoch", fontsize=11)
        axC.set_ylabel("Loss (MSE)", fontsize=11)
        axC.set_title("PINN Training (1D & 2D)", fontsize=13, fontweight="bold")
        axC.set_yscale("log")
        axC.grid(True, alpha=0.3, linestyle="--", which="both")
        axC.legend(fontsize=9)
    else:
        axC.text(0.5, 0.5, "No PINN logs found", ha="center", va="center")
        axC.axis("off")

    # ============================================================
    # Layout and save
    # ============================================================
    plt.suptitle(
        "Cantilever Support Optimization: Training Data & Convergence",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    out_path = FIG_DIR / "Cantilever_Story.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()


