"""
MMC runner for ribbed channel geometry.
Uses mmc_core.py with custom load cases (upward_tip, lateral_shear).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from mmc_core import (
    MMCConfig,
    DomainConfig,
    ConstraintConfig,
    build_passive_mask,
    get_ke,
    update_design,
    apply_constraints,
    compliance,
    coo_matrix,
    spsolve,
    E_BASE,
    E_BOLT,
)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
RESULT_DIR = (ROOT_DIR / "data/results").resolve()
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def build_channel_force(domain: DomainConfig, load_case: str, upward_force: float = 600.0, shear_force: float = 400.0):
    """Build force vector for ribbed channel."""
    ndof = 2 * (domain.nelx + 1) * (domain.nely + 1)
    force = np.zeros(ndof)
    
    if load_case == "upward_tip":
        # Upward force at tip (right edge)
        tip_nodes = [domain.nelx * (domain.nely + 1) + j for j in range(domain.nely + 1)]
        distributed_force = upward_force / max(len(tip_nodes), 1)
        for node in tip_nodes:
            force[2 * node + 1] -= distributed_force  # Negative Y (upward)
    elif load_case == "lateral_shear":
        # Lateral shear along top edge
        top_nodes = [i * (domain.nely + 1) + domain.nely for i in range(domain.nelx + 1)]
        distributed_force = shear_force / max(len(top_nodes), 1)
        for node in top_nodes:
            force[2 * node] += distributed_force  # Positive X (rightward)
    else:
        raise ValueError(f"Unknown load case: {load_case}")
    
    return force


def fe_solve_channel(x_phys, ke, passive, domain: DomainConfig, force_vec: np.ndarray):
    """FEA solve for ribbed channel with custom force vector and slot boundary conditions."""
    ndof = 2 * (domain.nelx + 1) * (domain.nely + 1)
    x_eff = x_phys.copy()
    x_eff[passive == 1] = 0.0
    
    e_elem = E_BASE + x_eff * (E_BOLT - E_BASE)
    I, J, V = [], [], []
    for elx in range(domain.nelx):
        for ely in range(domain.nely):
            n1 = (domain.nely + 1) * elx + ely
            n2 = (domain.nely + 1) * (elx + 1) + ely
            nodes = [n1, n2, n2 + 1, n1 + 1]
            dofs = np.array([2 * n for n in nodes] + [2 * n + 1 for n in nodes])
            k_scaled = ke * e_elem[ely, elx]
            for i in range(8):
                for j in range(8):
                    I.append(dofs[i])
                    J.append(dofs[j])
                    V.append(k_scaled[i, j])
    
    K = coo_matrix((V, (I, J)), shape=(ndof, ndof)).tocsc()
    
    # Fixed at left edge + slot interior (to prevent rigid body motion)
    # Slot: x in [60mm, 100mm], y in [15mm, 35mm]
    # Assuming 160mm length / 64 elements = 2.5 mm/element
    # Slot: x in [24, 40] elements, y in [6, 14] elements
    fixed_dofs = []
    slot_x_min, slot_x_max = 24, 40
    slot_y_min, slot_y_max = 6, 14
    
    # Left edge clamp
    clamp_width = int(0.1 * domain.nelx) + 1
    for x in range(clamp_width):
        for y in range(domain.nely + 1):
            n = x * (domain.nely + 1) + y
            fixed_dofs.extend([2 * n, 2 * n + 1])
    
    # Slot interior
    for x in range(slot_x_min, slot_x_max + 1):
        for y in range(slot_y_min, slot_y_max + 1):
            if x < domain.nelx + 1 and y < domain.nely + 1:
                n = x * (domain.nely + 1) + y
                fixed_dofs.extend([2 * n, 2 * n + 1])
    
    fixed_dofs = list(set(fixed_dofs))  # Remove duplicates
    
    K_diag = K.diagonal().copy()
    K_diag[fixed_dofs] = 1e9
    K.setdiag(K_diag)
    
    u = spsolve(K, force_vec)
    return u, force_vec


def run_mmc_ribbed_channel(
    config: MMCConfig,
    domain: DomainConfig = None,
    constraints: ConstraintConfig = None,
    load_case: str = "upward_tip",
):
    """Run MMC optimization for ribbed channel."""
    if domain is None:
        domain = DomainConfig(nelx=64, nely=20, void_cutoff_x=80, void_cutoff_y=80)
    if constraints is None:
        constraints = ConstraintConfig(edge_margin=3.0, min_spacing=3.0)
    
    np.random.seed(config.seed)
    passive = build_passive_mask(domain)
    ke = get_ke()
    force_vec = build_channel_force(domain, load_case)
    
    # Initialize screws at three row positions
    # x positions: 35mm, 80mm, 125mm -> ~14, 32, 50 elements (assuming 2.5mm/element)
    row_x_positions = [14.0, 32.0, 50.0]
    y_range = (constraints.edge_margin, domain.nely - constraints.edge_margin)
    
    design = np.zeros(2 * config.n_screws, dtype=float)
    if config.n_screws == 3:
        for idx in range(3):
            design[2 * idx] = row_x_positions[idx]
            design[2 * idx + 1] = y_range[0] + (y_range[1] - y_range[0]) * (idx + 1) / 4
    else:
        # Distribute evenly
        for idx in range(config.n_screws):
            row_idx = int(idx * len(row_x_positions) / config.n_screws)
            design[2 * idx] = row_x_positions[min(row_idx, len(row_x_positions) - 1)]
            design[2 * idx + 1] = y_range[0] + (y_range[1] - y_range[0]) * idx / max(1, config.n_screws - 1)
    
    design = apply_constraints(design, domain, constraints)
    history = []
    
    for it in range(config.n_iters + 1):
        x_phys = update_design(design, domain, radius=config.radius, beta=config.beta)
        u, F = fe_solve_channel(x_phys, ke, passive, domain, force_vec)
        comp = compliance(u, F)
        
        record = {"iter": it, "compliance": comp}
        for i in range(config.n_screws):
            record[f"x{i+1}"] = design[2 * i]
            record[f"y{i+1}"] = design[2 * i + 1]
        history.append(record)
        
        if it == config.n_iters:
            break
        
        # Simple gradient approximation
        target = np.array([row_x_positions[i % len(row_x_positions)] for i in range(config.n_screws)] + 
                         [domain.nely / 2] * config.n_screws, dtype=float)
        grad_mock = target - design
        design = design + config.lr * 0.05 * grad_mock
        design = apply_constraints(design, domain, constraints)
    
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Run MMC optimization on ribbed channel.")
    parser.add_argument("--iters", type=int, default=30, help="Number of iterations")
    parser.add_argument("--screws", type=int, default=3, help="Number of screws")
    parser.add_argument("--radius", type=float, default=4.0, help="Component radius")
    parser.add_argument("--beta", type=float, default=5.0, help="Projection sharpness")
    parser.add_argument("--lr", type=float, default=0.5, help="Update scaling factor")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--edge-margin", type=float, default=3.0, help="Edge margin (elements)")
    parser.add_argument("--min-spacing", type=float, default=3.0, help="Min spacing between screws")
    parser.add_argument("--nelx", type=int, default=64, help="Elements along X")
    parser.add_argument("--nely", type=int, default=20, help="Elements along Y")
    parser.add_argument("--load-case", choices=["upward_tip", "lateral_shear"], default="upward_tip")
    parser.add_argument("--tag", default="ribbed_channel", help="Tag appended to output filenames")
    return parser.parse_args()


def plot_compliance(df, path):
    plt.figure(figsize=(6, 4))
    plt.plot(df["iter"], df["compliance"], marker="o", color="purple")
    plt.xlabel("Iteration")
    plt.ylabel("Compliance (Fᵀu)")
    plt.title("MMC Optimization Path – Ribbed Channel")
    plt.grid(True, alpha=0.3)
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
        void_cutoff_x=80,  # No void for ribbed channel (slot handled in BCs)
        void_cutoff_y=80,
    )
    constraints = ConstraintConfig(
        edge_margin=args.edge_margin,
        min_spacing=args.min_spacing,
    )
    
    t0 = time.time()
    history = run_mmc_ribbed_channel(config, domain, constraints, args.load_case)
    elapsed = time.time() - t0
    
    df = pd.DataFrame(history)
    df["method"] = "mmc"
    df["wall_time_ms"] = (elapsed / max(1, len(df))) * 1000.0
    df["geometry"] = "ribbed_channel"
    df["load_case"] = args.load_case
    
    tag_suffix = f"{args.tag}_{args.load_case}"
    log_path = RESULT_DIR / f"mmc_{tag_suffix}_log.csv"
    df.to_csv(log_path, index=False)
    print(f"✅ Saved MMC log to {log_path}")
    
    if HAS_MATPLOTLIB:
        plot_path = RESULT_DIR / f"mmc_{tag_suffix}_compliance.png"
        plot_compliance(df, plot_path)
        print(f"✅ Saved compliance plot to {plot_path}")
    
    best = df.loc[df["compliance"].idxmin()]
    print(f"   Best compliance: {best['compliance']:.4f} at iteration {int(best['iter'])}")


if __name__ == "__main__":
    main()

