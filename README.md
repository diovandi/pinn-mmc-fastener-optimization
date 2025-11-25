# PINN vs MMC: Fastener Placement Optimization

A comparative study of Physics-Informed Neural Networks (PINNs) and Moving Morphable Components (MMC) for optimal fastener placement in structural optimization problems.

## Overview

This repository contains the implementation and analysis for a dual-method thesis comparing differentiable-physics PINNs against Moving Morphable Components (MMC) on fastener placement problems. The codebase follows a clean separation between code, data, documentation, and figures to ensure each workflow is easy to discover and maintain.

### Key Features

- **Approach A (PINN)**: Differentiable FEA kernels in Julia with PyTorch-based neural network training
- **Approach B (MMC)**: Moving Morphable Components topology optimization
- **Multi-geometry validation**: L-bracket, tapered plate, and ribbed channel geometries
- **Comprehensive analysis tools**: Unified comparison plots, Ansys export utilities, and generalization testing

## Repository Structure

- `src/approach_a_pinn/` – Julia differentiable FEA kernels, dataset generators, and the PyTorch training/inference stack (artifacts live in `src/approach_a_pinn/artifacts/`).
- `src/approach_b_mmc/` – MMC core module plus the CLI benchmark runner.
- `src/experiments/scenario_validation/` – Isolated tapered-plate validation workspace with its own `pinn/`, `mmc/`, `fea/`, `results/`, and `figures/` subfolders.
- `src/tools/` – Shared Python utilities (`Export_Ansys_Layouts.py`, `Generate_*` plotting scripts, `Test_Generalization.py`, etc.).
- `data/` – Authoritative store for CAD assets (`data/cad/`) and all logs/exports (`data/results/`).
- `docs/` – Human-facing material (`notes/`, `progress_reports/`, `proposals/`, `thesis_draft_chapters/`).
- `figures/` – Centralized PNG assets grouped by theme (`overview/`, `pinn/`, `mmc/`, `setup/`).
- `archive/` – Legacy experiments, scratch analyses, and historical notes kept for reference.

## Requirements

### Python Environment
- Python 3.8+
- PyTorch
- NumPy, SciPy, Matplotlib
- Jupyter (for notebooks)

### Julia Environment
- Julia 1.8+
- Required packages: See individual Julia scripts for dependencies

### Conda Environments
The project uses separate conda environments:
- `pinn_env` – For PINN training and inference
- `mmc_env` – For MMC optimization runs

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/diovandi/pinn-mmc-fastener-optimization.git
   cd pinn-mmc-fastener-optimization
   ```

2. Set up conda environments (if using conda):
   ```bash
   # Create and activate PINN environment
   conda create -n pinn_env python=3.9
   conda activate pinn_env
   pip install torch numpy scipy matplotlib jupyter
   
   # Create and activate MMC environment
   conda create -n mmc_env python=3.9
   conda activate mmc_env
   pip install numpy scipy matplotlib
   ```

3. Install Julia dependencies as needed for the specific scripts you plan to run.

## Usage

### Typical Workflows

### Approach A (Differentiable FEA + PINN)
1. Regenerate the multi-geometry dataset (L-bracket ×2 loads, tapered plate, ribbed channel ×2 loads):
   ```bash
   conda run -n pinn_env python src/experiments/scenario_validation/rollouts/run_all.py \
     --output-root data/results/multi_geom_training
   ```
2. Train or refresh the multi-geometry surrogate:
   ```bash
   conda run -n pinn_env python src/approach_a_pinn/train_multi_geom.py \
     --data-root data/results/multi_geom_training
   ```
3. Produce analysis artifacts / refreshed figures:
   ```bash
   conda run -n pinn_env python src/tools/Generate_Final_Plots.py
   conda run -n pinn_env python src/tools/Generate_Unified_Comparison.py
   ```
   Legacy single-geometry scripts (`GenerateData_Lshape_Batch.jl`, `TrainPINN.py`, `GenerateReportCharts.py`) are preserved for ablations but no longer represent the primary workflow.

### Approach B (MMC)
Run the CLI benchmark (saves logs to `data/results/`):

```bash
conda run -n mmc_env python src/approach_b_mmc/run_mmc_lbracket.py --tag lbracket
```

### Shared Utilities

- `src/tools/Generate_Unified_Comparison.py` – Rebuild convergence/speed/compliance plots directly from `data/results/`.
- `src/tools/Export_Ansys_Layouts.py` – Emit APDL-ready JSON payloads and templates to `data/results/ansys_exports/`.
- `src/tools/Test_Generalization.py` – Compare PINN predictions against MMC re-optimization on unseen load cases.

## Data, Figures, and Documentation

- All logs, manifests, and exports live in `data/results/` (e.g., `pinn_training_data.csv`, `lbracket_diff_fea_log.csv`, `mmc_runs_manifest.md`, `ansys_exports/`).
- CAD definitions are under `data/cad/` (e.g., `lbracket.json`).
- Publication-ready visuals are centralized in `figures/` for direct inclusion in slides or manuscripts.
- Writing assets sit in `docs/`:
  - `docs/notes/` – status explainers, implementation summary, planning notes.
  - `docs/progress_reports/` – dated supervisor updates.
  - `docs/proposals/` – thesis proposal artifacts.
  - `docs/thesis_draft_chapters/` – chapter drafts ready for editing.

## Scenario Validation Workspace

`src/experiments/scenario_validation/` now drives the **multi-geometry rollout plan**:
- `rollouts/` orchestrates Julia differentiable FEA runs for the L-bracket, tapered plate, and ribbed channel geometries (5 load cases total).
- `results/multi_geom_model_metrics.csv` captures MAE/MAPE per geometry/load after benchmarking the refreshed PINN against ground truth and the legacy model.
- `figures/multi_geom/model_mae_comparison.png` visualizes the improvement (≤12.4 J MAE vs >800 J legacy error off-distribution).

Use its README plus `dataset_plan.yaml` to adjust target sample counts (goal: ≥200 samples per load case before thesis submission).

## Status & Next Steps

- The multi-geometry PINN (Nov 25 2025) matches differentiable FEA within ≤1.1 % for all high-energy loads and ≤6.75 % for the tapered plate combined load, while keeping inference at ~0.009 ms/sample.
- Legacy single-geometry artifacts remain for regressions but should be cited as historical baselines only.
- Outstanding items:
  - Regenerate higher-volume datasets (≥200 samples/load) to shrink ribbed-channel shear error (currently 2.5 J absolute).
  - Execute Ansys validation via `data/results/ansys_exports/` using the refreshed layouts.
  - Propagate the new metrics/figures into the thesis chapters (`docs/thesis_draft_chapters/`) and status docs (`docs/notes/`, `docs/progress_reports/`).


## Contributing

This is a thesis research repository. For questions or collaboration inquiries, please open an issue or contact the repository owner.

## License

[Add your license here - e.g., MIT, Apache 2.0, or specify if this is academic/research code]

## Citation

If you use this work in your research, please cite:

```bibtex
[Add citation information when available]
```

## Author

- **Diovandi** – [GitHub](https://github.com/diovandi)

---

For detailed documentation, see the `docs/` directory. This README provides an overview; refer to individual component READMEs and documentation files for specific implementation details.

