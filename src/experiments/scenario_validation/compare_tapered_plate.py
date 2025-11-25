"""Generate geometry, PINN, and MMC comparison plots for the tapered plate."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCENARIO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = SCENARIO_ROOT / "results"
FIG_PINN = SCENARIO_ROOT / "figures" / "pinn"
FIG_MMC = SCENARIO_ROOT / "figures" / "mmc"
FIG_PINN.mkdir(parents=True, exist_ok=True)
FIG_MMC.mkdir(parents=True, exist_ok=True)

PINN_PATH = RESULTS_DIR / "pinn_new_part" / "pinn_tapered_plate_predictions.csv"
FEA_PATH = RESULTS_DIR / "fea_new_part" / "tapered_plate_diff_fea_log.csv"
MMC_PATH = RESULTS_DIR / "mmc_new_part" / "mmc_tapered_plate_log.csv"
METRICS_PATH = RESULTS_DIR / "tapered_plate_pinn_vs_fea_metrics.csv"

plt.style.use("seaborn-v0_8-whitegrid")


def _require(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing prerequisite file: {path}")


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (PINN_PATH, FEA_PATH, MMC_PATH):
        _require(path)
    pinn = pd.read_csv(PINN_PATH)
    fea = pd.read_csv(FEA_PATH)
    mmc = pd.read_csv(MMC_PATH)
    return pinn, fea, mmc


def merge_pinn_fea(pinn: pd.DataFrame, fea: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        pinn,
        fea,
        how="inner",
        left_on=["s1_x", "s1_y", "s2_x", "s2_y"],
        right_on=["s1_x", "s1_y", "s2_x", "s2_y"],
        suffixes=("_pinn", "_fea"),
    )
    merged["abs_error"] = (
        merged["pinn_predicted_compliance"] - merged["compliance"]
    ).abs()
    merged["rel_error_pct"] = 100.0 * merged["abs_error"] / merged["compliance"].clip(lower=1e-6)
    return merged


def plot_geometry_setup(output_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    rect = plt.Rectangle((0, 0), 140, 60, fill=False, linewidth=2.0, color="black")
    ax.add_patch(rect)
    grad = np.linspace(0.15, 0.65, 50)
    for idx, val in enumerate(grad):
        ax.fill_between(
            [idx * 140 / len(grad), (idx + 1) * 140 / len(grad)],
            0,
            60,
            color=str(val),
            alpha=0.15,
        )
    ax.annotate("Clamp", xy=(5, 30), xytext=(12, 45), arrowprops=dict(arrowstyle="->"))
    ax.annotate("Tip load 450 N", xy=(135, 48), xytext=(92, 75), arrowprops=dict(arrowstyle="->"))
    ax.annotate("Torsion 25 N-m", xy=(132, 12), xytext=(88, -5), arrowprops=dict(arrowstyle="->"))
    ax.text(20, 55, "Row A", fontsize=9, color="navy")
    ax.text(80, 55, "Row B", fontsize=9, color="navy")
    ax.set_xlim(-10, 150)
    ax.set_ylim(-20, 90)
    ax.set_aspect("equal")
    ax.set_title("Tapered Plate Geometry & Loads")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_pinn_vs_fea(merged: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    scatter = axes[0].scatter(
        merged["compliance"],
        merged["pinn_predicted_compliance"],
        c=merged["rel_error_pct"],
        cmap="viridis",
        edgecolor="k",
    )
    lims = [merged["compliance"].min(), merged["compliance"].max()]
    axes[0].plot(lims, lims, "r--", linewidth=1.0)
    axes[0].set_xlabel("FEA Compliance (J)")
    axes[0].set_ylabel("PINN Compliance (J)")
    axes[0].set_title("PINN vs FEA")
    fig.colorbar(scatter, ax=axes[0], label="Rel. Error (%)")

    axes[1].hist(merged["rel_error_pct"], bins=15, color="#4c78a8", alpha=0.85)
    axes[1].set_xlabel("Relative Error (%)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Error Distribution")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_error_maps(merged: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True, sharey=True)
    for idx, (ax, cols) in enumerate(zip(axes, [("s1_x", "s1_y"), ("s2_x", "s2_y")])):
        ax.scatter(
            merged[cols[0]],
            merged[cols[1]],
            c=merged["abs_error"],
            cmap="magma",
            edgecolor="k",
        )
        ax.set_title(f"Screw {idx + 1} error field")
        ax.set_xlabel("x (mm)")
        if idx == 0:
            ax.set_ylabel("y (mm)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_mmc_convergence(mmc_df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(mmc_df["iter"], mmc_df["compliance"], marker="o", linewidth=1.6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Compliance (a.u.)")
    ax.set_title("MMC Convergence – Tapered Plate")
    best_idx = mmc_df["compliance"].idxmin()
    ax.scatter(
        mmc_df.loc[best_idx, "iter"],
        mmc_df.loc[best_idx, "compliance"],
        color="red",
        zorder=4,
        label="Best",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_mmc_layout(mmc_df: pd.DataFrame, output_path: Path):
    best_idx = mmc_df["compliance"].idxmin()
    best_row = mmc_df.loc[best_idx]
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.add_patch(plt.Rectangle((0, 0), 56, 24, fill=False, linewidth=1.5))
    ax.scatter(
        [best_row["x1"], best_row["x2"]],
        [best_row["y1"], best_row["y2"]],
        s=120,
        c=["#1f77b4", "#ff7f0e"],
        edgecolor="k",
    )
    ax.set_title("MMC screw layout (grid coords)")
    ax.set_xlabel("nelx index")
    ax.set_ylabel("nely index")
    ax.set_xlim(0, 56)
    ax.set_ylim(0, 24)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_metrics(merged: pd.DataFrame):
    metrics = {
        "num_samples": len(merged),
        "mae_j": merged["abs_error"].mean(),
        "max_error_j": merged["abs_error"].max(),
        "mean_rel_pct": merged["rel_error_pct"].mean(),
        "p95_rel_pct": merged["rel_error_pct"].quantile(0.95),
    }
    pd.Series(metrics).to_csv(METRICS_PATH)


def main():
    pinn, fea, mmc_df = load_datasets()
    merged = merge_pinn_fea(pinn, fea)
    write_metrics(merged)

    plot_geometry_setup(FIG_PINN / "tapered_plate_setup.png")
    plot_pinn_vs_fea(merged, FIG_PINN / "pinn_vs_fea_tapered_plate.png")
    plot_error_maps(merged, FIG_PINN / "pinn_error_fields.png")
    plot_mmc_convergence(mmc_df, FIG_MMC / "mmc_convergence_tapered_plate.png")
    plot_mmc_layout(mmc_df, FIG_MMC / "mmc_layout_tapered_plate.png")

    mae = merged["abs_error"].mean()
    rel = merged["rel_error_pct"].mean()
    print(f"✅ Comparison figures saved. PINN MAE={mae:.2f} J ({rel:.2f} %)")


if __name__ == "__main__":
    main()

