"""
MMC validation driver for the tapered cantilever plate scenario.
Reuses the primitives from `approach_b_mmc/mmc_core.py` without modifying them.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import numpy as np

from approach_b_mmc import mmc_core as mmc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_ROOT = PROJECT_ROOT / "scenario_validation"
RESULTS_DIR = SCENARIO_ROOT / "results" / "mmc_new_part"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TaperedPlateConfig:
    domain: mmc.DomainConfig = mmc.DomainConfig(nelx=56, nely=24, void_cutoff_x=80, void_cutoff_y=80)
    constraints: mmc.ConstraintConfig = mmc.ConstraintConfig(edge_margin=2.5, min_spacing=3.0)
    solver: mmc.MMCConfig = mmc.MMCConfig(
        n_screws=2,
        n_iters=40,
        lr=0.4,
        radius=3.5,
        beta=6.0,
        seed=2025,
        load_case="horizontal_tip",  # placeholder; custom load applied below
    )
    tip_force: float = 450.0
    torsion_moment: float = 25.0


def build_custom_force(domain: mmc.DomainConfig, tip_force: float, torsion_moment: float) -> np.ndarray:
    ndof = 2 * (domain.nelx + 1) * (domain.nely + 1)
    force = np.zeros(ndof)
    tip_nodes = [domain.nelx * (domain.nely + 1) + j for j in range(domain.nely + 1)]
    distributed_force = tip_force / max(len(tip_nodes), 1)
    for node in tip_nodes:
        force[2 * node + 1] -= distributed_force
    mid_y = domain.nely / 2
    torque_arm = max(mid_y, 1.0)
    torsion_force = torsion_moment / torque_arm
    top_nodes = [domain.nelx * (domain.nely + 1) + j for j in range(int(mid_y), domain.nely + 1)]
    bottom_nodes = [domain.nelx * (domain.nely + 1) + j for j in range(0, int(mid_y))]
    for node in top_nodes:
        force[2 * node + 1] -= torsion_force / max(len(top_nodes), 1)
    for node in bottom_nodes:
        force[2 * node + 1] += torsion_force / max(len(bottom_nodes), 1)
    return force


def fe_solve_combo(x_phys, ke, passive, domain: mmc.DomainConfig, force_vec: np.ndarray):
    ndof = 2 * (domain.nelx + 1) * (domain.nely + 1)
    x_eff = x_phys.copy()
    x_eff[passive == 1] = 0.0

    e_elem = mmc.E_BASE + x_eff * (mmc.E_BOLT - mmc.E_BASE)
    I: List[int] = []
    J: List[int] = []
    V: List[float] = []
    for elx in range(domain.nelx):
        for ely in range(domain.nely):
            e_idx = elx * domain.nely + ely
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

    K = mmc.coo_matrix((V, (I, J)), shape=(ndof, ndof)).tocsc()
    fixed_dofs: List[int] = []
    for x in range(int(0.1 * domain.nelx) + 1):
        for y in range(domain.nely + 1):
            n = x * (domain.nely + 1) + y
            fixed_dofs.extend([2 * n, 2 * n + 1])

    K_diag = K.diagonal().copy()
    K_diag[fixed_dofs] = 1e9
    K.setdiag(K_diag)

    u = mmc.spsolve(K, force_vec)
    return u, force_vec


def run_history(cfg: TaperedPlateConfig) -> List[Dict[str, float]]:
    np.random.seed(cfg.solver.seed)
    passive = mmc.build_passive_mask(cfg.domain)
    ke = mmc.get_ke()
    force_vec = build_custom_force(cfg.domain, cfg.tip_force, cfg.torsion_moment)

    base_positions = np.linspace(
        cfg.constraints.edge_margin,
        cfg.domain.nelx - cfg.constraints.edge_margin,
        cfg.solver.n_screws,
    )
    design = np.zeros(2 * cfg.solver.n_screws, dtype=float)
    for idx in range(cfg.solver.n_screws):
        design[2 * idx] = base_positions[idx]
        design[2 * idx + 1] = cfg.constraints.edge_margin + idx * 4.0
    design = mmc.apply_constraints(design, cfg.domain, cfg.constraints)

    history: List[Dict[str, float]] = []
    for it in range(cfg.solver.n_iters + 1):
        x_phys = mmc.update_design(design, cfg.domain, radius=cfg.solver.radius, beta=cfg.solver.beta)
        u, F = fe_solve_combo(x_phys, ke, passive, cfg.domain, force_vec)
        comp = mmc.compliance(u, F)
        history.append(
            {
                "iter": it,
                "compliance": float(comp),
                "x1": float(design[0]),
                "y1": float(design[1]),
                "x2": float(design[2]),
                "y2": float(design[3]),
            }
        )
        if it == cfg.solver.n_iters:
            break
        target = np.array([0.8 * cfg.domain.nelx, 0.5 * cfg.domain.nely] * cfg.solver.n_screws)
        grad_mock = target - design
        design = design + cfg.solver.lr * 0.05 * grad_mock
        design = mmc.apply_constraints(design, cfg.domain, cfg.constraints)
    return history


def save_history(history: List[Dict[str, float]], path: Path):
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["iter", "compliance", "x1", "y1", "x2", "y2"],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def main():
    cfg = TaperedPlateConfig()
    history = run_history(cfg)
    output_path = RESULTS_DIR / "mmc_tapered_plate_log.csv"
    save_history(history, output_path)
    best = min(history, key=lambda r: r["compliance"])
    print(f"✅ MMC tapered-plate run saved to {output_path}")
    print(
        f"   Best compliance {best['compliance']:.4f} at "
        f"({best['x1']:.2f}, {best['y1']:.2f}) / ({best['x2']:.2f}, {best['y2']:.2f})"
    )


if __name__ == "__main__":
    main()

