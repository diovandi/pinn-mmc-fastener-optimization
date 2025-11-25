# Multi-Geometry PINN Validation Report

_Date:_ 25 Nov 2025  
_Artifacts:_  
- Dataset plan – `src/experiments/scenario_validation/dataset_plan.yaml`  
- Rollout scripts – `src/experiments/scenario_validation/rollouts/*.jl`  
- Combined datasets – `data/results/multi_geom_training/*.csv`  
- Multi-geometry PINN – `src/approach_a_pinn/artifacts_multi_geom/`  
- Benchmark metrics – `src/experiments/scenario_validation/results/multi_geom_model_metrics.csv`  
- Comparison plot – `src/experiments/scenario_validation/figures/multi_geom/model_mae_comparison.png`

---

## 1. Dataset Summary

| Geometry | Load cases | Samples (current) | Screw DOFs | Notes |
| --- | --- | --- | --- | --- |
| L-bracket | horizontal_tip, vertical_tip | 60 total (30 per load) | 2 screws (4 inputs) | Generated via `generate_lbracket_dataset.jl` using random placements constrained by the void. |
| Tapered plate | combined_tip | 40 | 2 screws (rows at x=20 & 80 mm) | Uses `NewPartBenchmark.jl`; compliance ≈ 22–24 J. |
| Ribbed channel (Part C) | upward_tip, lateral_shear | 80 total | 3 screws (rows at x=35/80/125 mm) | New mesh driver `RibbedChannelBenchmark.jl` with interior slot + dual load vectors. |

All datasets are stored under `data/results/multi_geom_training/` and are automatically regenerated via `rollouts/run_all.py`.

## 2. Training Configuration

- Script: `src/approach_a_pinn/train_multi_geom.py`
- Samples used: 180 (from the CSVs above)
- Model: 2-layer Tanh MLP with 96 hidden units (input dimension 6 screw DOFs + 3 geom one-hot + 5 load-case one-hot = 14 features).
- Optimizer/Loss: Adam, lr = 1e-3, MSE; 1 500 epochs.
- Final training loss ≈ 2.9e-4 (normalized space).
- Artifacts:
  - Weights: `src/approach_a_pinn/artifacts_multi_geom/pinn_multi_geom.pth`
  - Normalization stats: `.../norm_stats_multi_geom.npz`
  - Metadata (geometry/load mappings): `.../dataset_metadata.json`

## 3. Model Benchmarking (FEA Ground Truth vs PINNs)

Metrics reference `src/experiments/scenario_validation/results/multi_geom_model_metrics.csv`.

| Geometry / Load | Multi-PINN MAE (J) | Multi MAPE (%) | Legacy MAE (J) | Legacy MAPE (%) | Comments |
| --- | --- | --- | --- | --- | --- |
| L-bracket / horizontal | **3.28** | 0.95 | 905 | 279 | Multi-PINN captures both DOFs; legacy model off by orders of magnitude under horizontal loading. |
| L-bracket / vertical | **12.4** | 1.07 | 1 080 | 91 | Multi-PINN stays within ~1 % relative error despite mixed dataset. |
| Tapered plate / combined | **1.55** | 6.75 | 849 | 3 709 | Legacy PINN catastrophically extrapolates (>3 k% MAPE); new model succeeds thanks to new training data. |
| Ribbed channel / upward | **1.46** | 4.54 | n/a | n/a | New geometry validated; legacy PINN cannot handle three screws. |
| Ribbed channel / shear | **2.48** | 335 | n/a | n/a | Relative error inflated because compliance magnitude is <1 J; absolute error remains ~2.5 J. |

Inference latency averaged ~0.009 ms/sample for the multi-geometry model vs ~0.0016 ms for the lean legacy model; the speed trade-off is negligible relative to FEA.

## 4. Key Takeaways

1. **Generalization recovered:** The multi-geometry PINN delivers ≤1 % MAPE on both L-bracket load cases and remains within single-digit percentages on tapered plate and ribbed-channel upward loads.
2. **Legacy limitations exposed:** The original model fails whenever the geometry or load deviates from its training set (tapered plate / horizontal L-bracket). The new dataset + retraining step is mandatory before claiming cross-case inference.
3. **Channel shear case:** Large relative error stems from tiny ground-truth values (<1 J). Consider rescaling outputs or augmenting the training set with higher-force shear cases to shrink absolute error further.

## 5. Next Documentation Steps

- Reference this report plus the comparison figure in `project_status_explainer.md` and the thesis Methodology/Results chapters.
- Highlight that the multi-geometry model enables immediate inference on three benchmark parts, whereas the legacy PINN should only be cited for single L-bracket validation.
- Document residual risks: dataset size is still small (180 samples); plan to regenerate with the target counts from `dataset_plan.yaml` before final submission.

