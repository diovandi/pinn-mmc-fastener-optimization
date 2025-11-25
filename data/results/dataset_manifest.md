# PINN Training Dataset Manifest

## Dataset History

| Timestamp (UTC) | Script | Samples | Coverage |
|-----------------|--------|---------|----------|
| 2025-11-25 | `GenerateData_Lshape.jl` | 39 | Initial L-bracket rollouts (horizontal load only) |
| 2025-11-25 | `GenerateData_Lshape_Batch.jl` | 90 | Extended single-geometry dataset (horizontal + vertical tip) |
| 2025-11-25 | `src/experiments/scenario_validation/rollouts/run_all.py` | 180 | Multi-geometry corpus (L-bracket ×2 loads, tapered plate, ribbed channel ×2 loads) |

### Current Multi-Geometry Split (180 samples)
- **L-bracket:** 60 samples (30 horizontal tip, 30 vertical tip)
- **Tapered plate:** 40 samples (combined tip load)
- **Ribbed channel:** 80 samples (40 upward tip, 40 lateral shear)

## Artifacts
- `data/results/pinn_training_data.csv` — Legacy 90-sample L-bracket dataset (kept for regressions).
- `data/results/multi_geom_training/*.csv` — Authoritative per-geometry CSVs consumed by the latest training run.
- `src/approach_a_pinn/artifacts/pinn_model.pth` / `norm_stats.npz` — 90-sample checkpoint (legacy).
- `src/approach_a_pinn/artifacts_multi_geom/pinn_multi_geom.pth` / `norm_stats_multi_geom.npz` / `dataset_metadata.json` — Refreshed multi-geometry PINN artifacts (Nov 25, 2025).

## Reproduction
```
# Regenerate combined datasets (Julia rollouts orchestrated from Python)
conda run -n pinn_env python src/experiments/scenario_validation/rollouts/run_all.py

# Train the refreshed surrogate on all geometries / load cases
conda run -n pinn_env python src/approach_a_pinn/train_multi_geom.py \
  --data-root data/results/multi_geom_training
```

> Older single-geometry scripts (`GenerateData_Lshape*.jl`, `TrainPINN.py`) remain for ablation studies but should no longer be cited as the primary dataset pipeline.

