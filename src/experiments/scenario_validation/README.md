# Scenario Validation Workspace

This directory isolates the tapered cantilever plate validation so the primary PINN/MMC workflows remain untouched. Subfolders:

- `pinn/` – scripts that reuse the trained PINN for the new geometry and export predictions.
- `mmc/` – MMC configurations and drivers for the new part/load case.
- `fea/` – Julia-based differentiable FEA definitions mirroring the L-bracket benchmark, adapted to the tapered plate.
- `results/` – Subdirectories (`pinn_new_part/`, `fea_new_part/`, `mmc_new_part/`) storing CSV/HDF5 logs for each method.
- `figures/` – Visualization outputs split into `pinn/` and `mmc/` suites that mirror the existing L-bracket figures.
- `rollouts/` – Automation scripts that generate large multi-geometry datasets (`generate_*_dataset.jl`, `run_all.py`).

All scripts here import shared utilities via relative paths within `src/` (e.g., `../../approach_a_pinn`), so no existing training or optimization code needs modification.

## Multi-Geometry Dataset Roadmap

- `dataset_plan.yaml` captures the roll-out specifications for three geometries:
  - The legacy L-bracket (horizontal & vertical tip loads).
  - The tapered cantilever plate (combined force + torsion).
  - A new ribbed channel part (Part C) with two distinct load cases and three screw rows.
- Each entry records mesh drivers, boundary conditions, screw sampling domains, and target sample counts so the upcoming automation scripts can parse and execute the plan consistently.
- `rollouts/run_all.py` parses the plan and invokes the appropriate Julia dataset generators. Use `--dry-run` to inspect commands, or `--geometry <name>` to run a single geometry.

