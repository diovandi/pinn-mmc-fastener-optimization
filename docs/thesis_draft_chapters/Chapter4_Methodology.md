# Chapter 4: Methodology

## 4.1 Implementation Overview

### 4.1.1 Software Stack

**Approach A (PINN):**
- Julia 1.12 with Zygote.jl for automatic differentiation
- PyTorch 2.0+ for neural network training
- Conda environment: `pinn_env`

**Approach B (MMC):**
- Python 3.10 with NumPy, SciPy
- Matplotlib for visualization
- Conda environment: `mmc_env`

### 4.1.2 Shared Infrastructure

- Common geometry: `data/cad/lbracket.json`
- Unified logging format: CSV with method, iteration, compliance, coordinates, wall-time
- Results directory: `data/results/`

## 4.2 Approach A Implementation

### 4.2.1 Differentiable FEA Solver

**File:** `src/approach_a_pinn/DiffFEA_2D.jl`

- CST (Constant Strain Triangle) elements
- Sparse matrix assembly
- Penalty method for boundary conditions
- Matrix damping (1e-6) for numerical stability

### 4.2.2 Data Generation

**Files:** `src/experiments/scenario_validation/rollouts/*.jl`, orchestrated via `rollouts/run_all.py`

- Generates **multi-geometry datasets**: L-bracket (horizontal & vertical), tapered plate (combined), ribbed channel (upward & shear).
- Current corpus: 180 samples (60/40/80 split). Target per `dataset_plan.yaml`: ≥200 samples per load.
- Outputs CSVs under `data/results/multi_geom_training/` plus metadata (`dataset_metadata.json`) and manifest entries (`data/results/dataset_manifest.md`).
- Legacy script `GenerateData_Lshape_Batch.jl` remains for ablations (single-geometry 90-sample set).

### 4.2.3 PINN Training

**File:** `src/approach_a_pinn/train_multi_geom.py`

- Architecture: 14-input MLP (6 screw DOFs + geometry/load one-hot) with two 96-neuron Tanh layers → 1 output.
- Training: Adam (lr = 1e-3), 1 500 epochs, batch size derived from dataset size; final normalized loss ≈2.9e-4.
- Artifacts: `src/approach_a_pinn/artifacts_multi_geom/pinn_multi_geom.pth`, `norm_stats_multi_geom.npz`, and `dataset_metadata.json`.
- Legacy `TrainPINN.py` checkpoint (`artifacts/pinn_model.pth`) archived for regressions.

### 4.2.4 Benchmark Logging

**File:** `src/approach_a_pinn/LBracketBenchmarkLogger.jl`

- Runs optimization with logging
- Records: iteration, compliance, screw coordinates, wall-time
- Output: `data/results/lbracket_diff_fea_log.csv`

## 4.3 Approach B Implementation

### 4.3.1 MMC Core Module

**File:** `src/approach_b_mmc/mmc_core.py`

- Configurable via dataclasses: `DomainConfig`, `ConstraintConfig`, `MMCConfig`
- Component projection with level set functions
- FEA solve with scipy.sparse
- Constraint enforcement: spacing and edge margins

### 4.3.2 Benchmark Runner

**File:** `src/approach_b_mmc/run_mmc_lbracket.py`

- Command-line interface for parameter configuration
- Supports multiple load cases: `horizontal_tip`, `vertical_tip`
- Generates: compliance plots, component trajectory plots
- Output: `data/results/mmc_log_<tag>.csv`

### 4.3.3 Configuration Hooks

All parameters exposed via argparse:
- Domain: `--nelx`, `--nely`, `--void-x`, `--void-y`
- Constraints: `--edge-margin`, `--min-spacing`
- Optimization: `--iters`, `--radius`, `--beta`, `--lr`
- Load case: `--load-case`

## 4.4 Validation Pipeline

### 4.4.1 Ansys Export

**File:** `src/tools/Export_Ansys_Layouts.py`

- Exports optimized screw positions to JSON
- Creates Ansys APDL script template
- Generates validation summary with target error thresholds

### 4.4.2 Comparative Analysis

**File:** `src/tools/Generate_Unified_Comparison.py`

- Overlays convergence plots from both methods
- Speed comparison bar charts
- Final compliance comparison tables

### 4.4.3 Generalization Testing

**File:** `src/tools/Test_Generalization.py`

- Tests PINN on unseen load cases
- Compares with MMC re-optimization results
- Generates generalization error analysis

## 4.5 Reproducibility

All scripts include:
- Random seed control
- Version tracking via manifests (`dataset_manifest.md`, `mmc_runs_manifest.md`)
- Normalized data formats
- Clear parameter documentation

