# Chapter 5: Results

## 5.1 Approach A: PINN Results

### 5.1.1 Dataset Characteristics

- **Total Samples:** 180 (multi-geometry corpus generated via `src/experiments/scenario_validation/rollouts/run_all.py`).
  - L-bracket: 60 samples (30 horizontal tip, 30 vertical tip).
  - Tapered plate: 40 samples (combined tip load).
  - Ribbed channel: 80 samples (40 upward tip, 40 lateral shear).
- **Feature vector:** 6 screw DOFs + 3 geometry one-hot + 5 load one-hot → 14 inputs.
- **Storage:** `data/results/multi_geom_training/*.csv` with metadata recorded in `dataset_metadata.json` and summarized in `data/results/dataset_manifest.md`.

### 5.1.2 Surrogate Accuracy (vs Differentiable FEA)

| Geometry / Load | MAE (J) | MAPE (%) | Legacy PINN Error |
|-----------------|---------|----------|-------------------|
| L-bracket / Horizontal tip | **3.28** | 0.95 | 905 J |
| L-bracket / Vertical tip | **12.4** | 1.07 | 1 080 J |
| Tapered plate / Combined tip | **1.55** | 6.75 | 849 J |
| Ribbed channel / Upward tip | **1.46** | 4.54 | n/a |
| Ribbed channel / Lateral shear | **2.48** (absolute) | 335* | n/a |

*Relative error is large only because the ground-truth compliance is <1 J; absolute error remains ~2.5 J. The dataset plan targets ≥200 samples/load to bring this below 1 J.

**Training configuration:** 2×96 Tanh MLP, Adam (lr = 1e-3), 1 500 epochs, final normalized loss ≈2.9e-4. Artifacts live in `src/approach_a_pinn/artifacts_multi_geom/`.

### 5.1.3 Speed Benchmark

- **Julia Diff-FEA:** 10.4 ms per iteration (baseline physics solve).
- **Multi-geometry PINN:** ~0.009 ms per inference.
- **Legacy PINN:** 0.047 ms per inference (4-input MLP retained for ablations).
- **Speedup:** ≥1 000× acceleration (diff-FEA vs multi-geometry PINN) while preserving ≤1.1 % error on high-energy loads.

### 5.1.4 Optimization Trajectory

- Differentiable FEA optimizer still reduces compliance from ~976 J to 939.6 J over 26 iterations; these trajectories seed the L-bracket portion of the dataset.
- Replacing the FEA call with the multi-geometry PINN preserves convergence behavior while cutting wall-clock time from 0.27 s (FEA) to milliseconds (PINN) for a 26-iteration run.

## 5.2 Approach B: MMC Results

### 5.2.1 Convergence Behavior

- **Iterations:** 30 iterations to convergence
- **Final Compliance:** 0.426 (normalized) for horizontal tip load case
- **Time per Iteration:** 118 ms (Python implementation)

### 5.2.2 Component Trajectories

Screw components migrate from initial positions toward the high-strain inner corner of the L-bracket, as expected from compliance minimization. Trajectory plots (`mmc_paths_lbracket.png`) show smooth convergence.

### 5.2.3 Load Case Variation

MMC was tested on two load cases:
- **Horizontal tip:** Final compliance = 0.426
- **Vertical tip:** Final compliance = 0.760

Both cases required full re-optimization, demonstrating MMC's load-case-specific behavior.

## 5.3 Comparative Summary

| Metric | Diff-FEA | Multi-Geometry PINN | MMC |
|--------|----------|---------------------|-----|
| Time/Iteration | 10.4 ms | 0.009 ms | 118 ms |
| Final Compliance (L-bracket) | 939.6 J | Matches diff-FEA within ≤1 % | 0.426 (normalized) |
| Dataset / Training | N/A | 180 samples, ~2 h training | N/A |
| Generalization | N/A | Immediate inference across 3 geometries / 5 loads (≤6.75 % MAPE except shear) | Requires full re-optimization per load |

## 5.4 Key Findings

1. **Cross-geometry accuracy restored:** Multi-geometry PINN maintains ≤1.1 % MAPE on L-bracket loads and single-digit percentages on tapered/ribbed upward loads; only the ribbed-channel shear case awaits more data.
2. **Speed advantage preserved:** Even with the wider architecture, inference remains ~0.009 ms/sample (>1 000× faster than diff-FEA, >13 000× faster than MMC).
3. **MMC remains interpretable:** It still provides explicit component trajectories and constraint enforcement, making it the most transparent option for single-load studies.
4. **Next steps:** Scale each load case to ≥200 samples, run MMC on the new geometries, and execute Ansys/CalculiX validation to finalize the comparative story.

