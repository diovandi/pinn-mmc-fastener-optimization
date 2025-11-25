# Unified Figure Inventory & Results Log

This document maps all generated figures and data artifacts to their target thesis sections, following the plan outlined in `help me with the next steps.md`.

## Approach A (PINN) Figures

| Figure File | Input Data | Key Values | Message | Target Section |
|-------------|------------|------------|---------|----------------|
| `figures/overview/Proof_Speedup.png` | `lbracket_diff_fea_log.csv` | FEA: ~10.4 ms/iter, legacy PINN: 0.047 ms/iter, multi-geometry PINN: 0.009 ms/iter | Surrogate inference remains ≥1e3× faster than FEA even after widening the model | Ch. 5 §5.2.1 (Speed Benchmark) |
| `figures/multi_geom/model_mae_comparison.png` | `src/experiments/scenario_validation/results/multi_geom_model_metrics.csv` | MAE ≤12.4 J on all high-energy loads, ≤6.75 % MAPE on tapered plate | Multi-geometry PINN matches FEA across three parts while legacy model collapses off-distribution | Ch. 5 §5.2.3 & Ch. 6 §6.3 (Generalization) |
| `figures/overview/Proof_Distribution.png` | `pinn_training_data.csv` (legacy) | 90 samples, range 544–1 509 J | Historical dataset coverage (kept for appendix) | Appendix / Legacy comparison |
| `data/results/dataset_histogram.png` | `data/results/pinn_training_data.csv` | Compliance distribution histogram | Legacy dataset visualization | Ch. 4 (legacy ablation) |
| `data/results/dataset_visualization.png` | `pinn_training_data.csv` | Screw placement scatter on L-bracket | Spatial distribution baseline | Ch. 4 |
| `figures/pinn/training_curve.png` | `TrainPINN.py` logs | Final MSE <0.01 | Legacy training convergence | Appendix |

> **Update pending:** regenerate histogram/visualization with `multi_geom_training/*.csv` once ≥200 samples/load are available. Current figures stay in appendix for historical context.

## Approach B (MMC) Figures

| Figure File | Input Data | Key Values | Message | Target Section |
|-------------|------------|------------|---------|----------------|
| `mmc_lbracket_compliance.png` | `mmc_lbracket_log.csv` | Final compliance: ~0.43 (normalized), 30 iterations | MMC convergence on L-bracket benchmark | Chapter 5, Section 5.3.1 (MMC Optimization Path) |
| `mmc_paths_lbracket.png` | `mmc_lbracket_log.csv` | Component trajectories from initial to final positions | Geometric component migration visualization | Chapter 5, Section 5.3.2 (Component Trajectories) |
| `mmc_compliance_lbracket.png` | `mmc_lbracket_log.csv` | Compliance vs iteration curve | MMC optimization convergence | Chapter 5, Section 5.3.1 (MMC Results) |
| `mmc_compliance_vertical.png` | `mmc_log_vertical.csv` | Alternative load case results | MMC generalization to different load cases | Chapter 5, Section 5.3.3 (Load Case Variation) |
| `mmc_paths_vertical.png` | `mmc_log_vertical.csv` | Component paths for vertical load case | Load case sensitivity visualization | Chapter 5, Section 5.3.3 (Load Case Variation) |

## Comparative Analysis Figures

| Figure File | Input Data | Key Values | Message | Target Section |
|-------------|------------|------------|---------|----------------|
| `data/results/comparison_compliance.png` | `method_comparison.csv` | PINN vs MMC final compliance (legacy baseline) | Historical single-geometry comparison | Appendix |
| `data/results/comparison_time.png` | `method_comparison.csv` | Legacy PINN 0.047 ms vs MMC 118 ms | Historical runtime trade-off | Appendix |
| `data/results/unified_convergence_comparison.png` | `lbracket_diff_fea_log.csv`, `mmc_lbracket_log.csv` | Side-by-side convergence | Updated with annotations referencing multi-geometry status | Ch. 6 §6.1 |
| `data/results/unified_speed_comparison.png` | Timing measurements | Diff-FEA 10.4 ms, Legacy PINN 0.047 ms, Multi PINN 0.009 ms, MMC 118 ms | Holistic runtime view | Ch. 6 §6.1 |
| `data/results/unified_compliance_comparison.png` | `method_comparison.csv` + new callouts | MMC normalized vs PINN Joule values | Solution quality context | Ch. 6 §6.2 |
| `figures/overview/L_Bracket_Setup_Visual.png` | `lbracket.json` | Benchmark geometry | Ch. 3 §3.1 |

## Data Artifacts

| File | Description | Used In |
|------|-------------|---------|
| `data/results/multi_geom_training/*.csv` | Authoritative per-geometry datasets (L-bracket, tapered plate, ribbed channel) | Multi-geometry training, Ch. 4 dataset section |
| `data/results/pinn_training_data.csv` | Legacy 90-sample L-bracket dataset | Historical comparison / appendix |
| `data/results/lbracket_diff_fea_log.csv` | 26-iteration diff-FEA optimization | Speed benchmarks |
| `data/results/mmc_lbracket_log.csv`, `mmc_log_vertical.csv` | MMC convergence logs | Ch. 5 MMC |
| `data/results/method_comparison.csv` | Aggregated metrics (legacy) | Unified plots (annotated with refreshed context) |
| `src/experiments/scenario_validation/results/multi_geom_model_metrics.csv` | New MAE/MAPE table for PINNs | Ch. 5 & Ch. 6 generalization |
| `src/approach_a_pinn/artifacts_multi_geom/*` | Multi-geometry PINN weights + stats + metadata | Reproducibility appendix |

## Notes

- All compliance values for MMC are normalized (0–1 scale) and need conversion to physical units (J) for direct comparison
- PINN training used 90 samples; original target was 80–100, so this meets the requirement
- MMC runs include both baseline (lbracket) and exploratory (vertical) load cases
- Figure numbering will be finalized during thesis drafting (Chapters 5–7)

## Multi-Geometry Figures

| Figure File | Input Data | Key Values | Message | Target Section |
|-------------|------------|------------|---------|----------------|
| `figures/multi_geom/model_mae_comparison.png` | `multi_geom_model_metrics.csv` | MAE ≤12.4 J vs legacy >800 J | Proof that refreshed dataset fixes off-distribution collapse | Ch. 5 & 6 |
| `src/experiments/scenario_validation/figures/multi_geom/dataset_split.png` (TODO) | `dataset_metadata.json` | 60 / 40 / 80 sample breakdown | Visual summary of multi-geometry corpus | Ch. 4 once regenerated |
| `generalization_test_results.png` | `pinn_generalization_test.csv`, `mmc_load_case_comparison.csv` | Updated callouts referencing multi-geometry PINN as new baseline | Ch. 6 §6.3 |

## Next Actions

- [x] Add multi-geometry MAE comparison figure
- [ ] Regenerate dataset distribution/coverage plots using the 180-sample corpus (blocked on ≥200 samples/load goal)
- [ ] Execute Ansys validation and add residual plots
- [ ] Export updated MMC vs PINN overlay including ribbed/tapered contexts
- [ ] Finalize figure captions with numeric callouts in thesis chapters

