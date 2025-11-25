import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mmc_core import (
    MMCConfig,
    DomainConfig,
    ConstraintConfig,
    run_mmc_lbracket,
)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
RESULT_DIR = (ROOT_DIR / "data/results").resolve()
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the MMC L-bracket benchmark with configurable parameters."
    )
    parser.add_argument("--iters", type=int, default=30, help="Number of iterations")
    parser.add_argument("--screws", type=int, default=2, help="Number of screws")
    parser.add_argument("--radius", type=float, default=4.0, help="Component radius")
    parser.add_argument("--beta", type=float, default=5.0, help="Projection sharpness")
    parser.add_argument("--lr", type=float, default=0.5, help="Update scaling factor")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--edge-margin", type=float, default=1.0, help="Edge margin (elements)")
    parser.add_argument("--min-spacing", type=float, default=2.0, help="Min spacing between screws")
    parser.add_argument("--nelx", type=int, default=40, help="Elements along X")
    parser.add_argument("--nely", type=int, default=40, help="Elements along Y")
    parser.add_argument("--void-x", type=int, default=16, help="Void cutoff along X")
    parser.add_argument("--void-y", type=int, default=16, help="Void cutoff along Y")
    parser.add_argument("--load-case", choices=["horizontal_tip", "vertical_tip"], default="horizontal_tip")
    parser.add_argument("--tag", default="lbracket", help="Tag appended to output filenames")
    return parser.parse_args()


def plot_compliance(df, path):
    plt.figure(figsize=(6, 4))
    plt.plot(df["iter"], df["compliance"], marker="o", color="purple")
    plt.xlabel("Iteration")
    plt.ylabel("Compliance (Fᵀu)")
    plt.title("MMC Optimization Path – L-Bracket")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_trajectories(df, path):
    plt.figure(figsize=(6, 4))
    for screw_idx in [1, 2]:
        plt.plot(df["iter"], df[f"x{screw_idx}"], label=f"Screw {screw_idx} X")
        plt.plot(df["iter"], df[f"y{screw_idx}"], label=f"Screw {screw_idx} Y", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("Coordinate (elements)")
    plt.title("Component Trajectories")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    config = MMCConfig(
        n_screws=args.screws,
        n_iters=args.iters,
        lr=args.lr,
        radius=args.radius,
        beta=args.beta,
        seed=args.seed,
        load_case=args.load_case,
    )
    domain = DomainConfig(
        nelx=args.nelx,
        nely=args.nely,
        void_cutoff_x=args.void_x,
        void_cutoff_y=args.void_y,
    )
    constraints = ConstraintConfig(
        edge_margin=args.edge_margin,
        min_spacing=args.min_spacing,
    )

    t0 = time.time()
    history = run_mmc_lbracket(config, domain, constraints)
    elapsed = time.time() - t0

    df = pd.DataFrame(history)
    df["method"] = "mmc"
    df["wall_time_ms"] = (elapsed / max(1, len(df))) * 1000.0
    df["edge_margin"] = args.edge_margin
    df["min_spacing"] = args.min_spacing
    df["load_case"] = args.load_case

    suffix = f"_{args.tag}"
    csv_path = RESULT_DIR / f"mmc_log{suffix}.csv"
    compliance_plot = RESULT_DIR / f"mmc_compliance{suffix}.png"
    traj_plot = RESULT_DIR / f"mmc_paths{suffix}.png"

    df.to_csv(csv_path, index=False)
    print(f"Saved MMC log to {csv_path}")
    print(f"Total time: {elapsed:.3f}s ({elapsed/len(df):.4f}s per iter)")

    plot_compliance(df, compliance_plot)
    plot_trajectories(df, traj_plot)
    print(f"Saved compliance plot to {compliance_plot}")
    print(f"Saved trajectory plot to {traj_plot}")


if __name__ == "__main__":
    main()

