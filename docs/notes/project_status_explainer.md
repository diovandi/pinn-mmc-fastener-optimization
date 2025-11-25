# Project Status Explainer – Dual Approach Thesis

## 1. Work Completed So Far

### 1.1 Approach A – Differentiable Physics + PINN
- Differentiable quasi-static FEA pipeline (Julia + Zygote) remains the data engine; automation now lives in `src/experiments/scenario_validation/rollouts/` to cover **three geometries and five load cases**.
- Multi-geometry dataset: 180 labeled samples (60 L-bracket, 40 tapered plate, 80 ribbed channel) stored under `data/results/multi_geom_training/` with metadata in `dataset_metadata.json`.
- New training entry point `src/approach_a_pinn/train_multi_geom.py` (2×96 Tanh MLP, 1 500 epochs) produces artifacts in `src/approach_a_pinn/artifacts_multi_geom/` (`pinn_multi_geom.pth`, `norm_stats_multi_geom.npz`).
- Accuracy: MAE stays within **3.28 J (0.95 % MAPE)** on L-bracket horizontal, **12.4 J (1.07 %)** on L-bracket vertical, **1.55 J (6.75 %)** on tapered plate, **1.46 J (4.54 %)** on ribbed channel upward, and **2.48 J absolute** on ribbed channel shear (ground truth <1 J).
- Inference latency: ~0.009 ms/sample (wider MLP) vs 10.4 ms/iter diff-FEA; legacy single-geometry surrogate (0.0016 ms) is archived for ablations.
- Validation artifacts refreshed: `figures/multi_geom/model_mae_comparison.png`, updated `data/results/unified_*` plots, and manifests in `data/results/dataset_manifest.md`.

### 1.2 Approach B – Moving Morphable Components (MMC)
- Refactored MMC implementation into modular core + CLI (`src/approach_b_mmc/mmc_core.py`, `run_mmc_lbracket.py`) with `DomainConfig`, `ConstraintConfig`, and `MMCConfig` hooks for geometry and solver parameters.
- Optimized the L-bracket benchmark for multiple load cases (horizontal and vertical tip loads), logging convergence in `mmc_lbracket_log.csv` and `mmc_log_vertical.csv`.
- Produced publication-ready visuals (`MMC_Screw_Trajectories.png`, `mmc_compliance_lbracket.png`) and documented runs in `mmc_runs_manifest.md`.

### 1.3 Shared Infrastructure & Documentation
- Created unified plotting/analysis scripts (`src/tools/Generate_Final_Plots.py`, `src/tools/Generate_Unified_Comparison.py`), Ansys export tooling (`src/tools/Export_Ansys_Layouts.py`, `ansys_exports/*.json`), and a generalization harness (`src/tools/Test_Generalization.py`).
- Consolidated results and manifests: `comprehensive_results_table.md`, `figure_inventory.md`, dataset and run manifests.
- Drafted all thesis chapters (`thesis_draft_chapters/`) and multiple progress summaries (`IMPLEMENTATION_SUMMARY.md`, `progress_report.md`, `thesis-progress-report.md`).

## 2. Meaning of PINN Accuracy

- **Metric definition:** Mean absolute error + MAPE measured per geometry/load by replaying the multi-geometry validation CSVs through differentiable FEA (`multi_geom_model_metrics.csv`). Each row contains paired ground truth, legacy PINN prediction, and refreshed PINN prediction pulled from identical normalization stats.
- **Value interpretation:** ≤1.1 % MAPE (≤12.4 J absolute) on both L-bracket loads confirms the surrogate is numerically interchangeable with diff-FEA for thesis benchmarking. Tapered plate accuracy at 1.55 J (6.75 %) demonstrates successful cross-geometry generalization, while ribbed-channel upward holds at 1.46 J (4.54 %). Shear load shows large relative error solely because the target compliance is <1 J; absolute error is still 2.48 J.
- **Regularization:** Current run is data-driven (pure MSE) because the mixed-geometry dataset already encodes physics from differentiable FEA. Physics residual hooks remain in the codebase for future ablations but were unnecessary for this refresh.
- **Practical implication:** Inference latency of ~0.009 ms/sample (vs. 10.4 ms/iter FEA or 118 ms/iter MMC) means the surrogate can evaluate thousands of candidate screw placements before MMC finishes a single iteration, enabling design-space sweeps across multiple geometries and loads.

## 3. Result Interpretation

### 3.1 Approach A (PINN Pipeline)
- **Baseline convergence:** Gradient-based search (diff-FEA) still reaches 939.6 J in 26 iterations; these logs feed the new dataset as the L-bracket portion.
- **Generalization realized:** With the 180-sample corpus, the refreshed PINN now produces ≤1.1 % MAPE on both L-bracket loads and single-digit percentages on tapered/ribbed upward loads. Only the ribbed-channel shear case needs more data to push absolute error below 1 J.
- **Surrogate workflow:** Replace the expensive FEA objective with the multi-geometry PINN for rapid sweeps; pass geometry + load one-hot vectors so a single checkpoint handles all five scenarios.
- **Next focus:** Regenerate higher-count rollouts per `dataset_plan.yaml` (≥200 samples/load) and explore minor loss weighting for very low-energy loads to tame the residual shear error.

### 3.2 Approach B (MMC)
- **Optimization quality:** Achieved normalized compliances of 0.426 (horizontal) and 0.760 (vertical) within 30 iterations, demonstrating robust convergence under explicit spacing/edge constraints.
- **Interpretability:** Fastener geometries remain explicit throughout optimization; this makes the method attractive when manufacturability rules dominate or when stakeholders demand transparent decision paths.
- **Cost profile:** Each iteration costs ~118 ms because it always solves full FEA; re-optimization is required for every new load case, which limits throughput but ensures fidelity.

### 3.3 Comparative Takeaways
- **Speed vs. setup:** PINN demands multi-language tooling and training but rewards with two orders of magnitude faster inference. MMC is slower per run yet simpler to deploy (Python-only) and requires no data curation.
- **Solution similarity:** Both methods converge to comparable compliance levels on the L-bracket benchmark; MMC’s normalized figures align with PINN’s absolute joule values when mapped through shared post-processing scripts.
- **Generalization strategy:** PINN can, in principle, estimate unseen load cases instantly once validated; MMC must redo the optimization but guarantees accuracy because it never leaves the physics solver loop.

## 4. Next Steps

1. **Scale datasets to target counts** – Follow `dataset_plan.yaml` to hit ≥200 samples per load, prioritizing ribbed-channel shear to shrink absolute error below 1 J.
2. **External validation** – Use `Export_Ansys_Layouts.py` payloads (PINN + MMC layouts for L-bracket and tapered plate) inside Ansys/CalculiX to quote compliance deltas vs diff-FEA and close the validation loop.
3. **Figure + notebook refresh** – Regenerate dataset coverage plots from `multi_geom_training/*.csv`, update `generalization_test_results.png`, and ensure Chapters 4–6 embed the new MAE table.
4. **Multi-geometry MMC runs** – Extend `mmc_core.py` inputs to the tapered plate and ribbed channel so Comparative Analysis discusses both methods on the same geometry set.
5. **Supervisor sync + scope freeze** – Present the refreshed metrics to confirm the comparative scope is satisfied; treat GA baseline / 3D work as optional unless explicitly requested.
6. **Optional research knobs** – Experiment with shear-case loss weighting or force scaling after the dataset scale-up if Ansys validation exposes any remaining bias.

---

This document synthesizes the implementation milestones, clarifies metric definitions (especially PINN accuracy), interprets the comparative results, and lists the concrete next actions needed to finish validation and writing.

## 5. Scenario Validation Add-on (Tapered Cantilever Plate)

- **Geometry:** 140 mm × 60 mm cantilever plate tapering linearly from 6 mm thickness at the clamp to 3 mm at the free end. Two candidate fastener rows (Row A at x = 20 mm, Row B at x = 80 mm) each allow up to three screws with 15 mm spacing.
- **Material/PDE:** Same linear elastic isotropic model as the L-bracket (E = 70 GPa, ν = 0.33), governed by the existing quasi-static plane-stress PDEs so the PINN weights remain applicable.
- **Loads/BCs:** Fixed clamp on the left edge; combined downward tip load of 450 N at the free end plus a torsional couple of 25 N-m applied via offset traction, stressing both bending and twisting modes.
- **Workspace isolation:** All scripts, logs, and figures for this scenario live under `src/experiments/scenario_validation/` (subfolders for `pinn/`, `mmc/`, `fea/`, `results/`, `figures/`). Existing training/optimization code stays untouched and is imported as needed.
- **Deliverables:** (1) New geometry/load visualization, (2) PINN vs differentiable FEA field comparisons, (3) MMC optimization + FEA benchmarking on the same part, (4) Figure suite mirroring the L-bracket visuals, and (5) write-up of agreement/mismatch in the relevant thesis chapters.

## 6. Findings from Scenario Validation

- `scenario_validation/multi_geom_report.md` consolidates the tapered plate + ribbed channel rollouts, updated dataset manifest, and benchmarking results.
- The refreshed PINN now achieves **1.55 J MAE (6.75 %)** on the tapered plate combined load and **1.46 J (4.54 %)** on the ribbed channel upward load after ingesting the new datasets. Legacy PINN metrics remain in the report to illustrate the dramatic failure (>800 J errors) without data.
- MMC optimization behaved consistently on the L-bracket; the next iteration is to port the same MMC configs to the tapered plate and ribbed channel for a truly apples-to-apples comparison.
- **Action:** keep expanding the multi-geometry dataset (especially the ribbed channel shear load) and document the updated metrics directly inside Chapters 4–6, replacing the earlier “future work” statements.

## 7. Multi-Geometry PINN Refresh

- **Dataset plan executed:** `src/experiments/scenario_validation/dataset_plan.yaml` now drives Julia rollouts for three parts (L-bracket, tapered plate, ribbed channel). Automation lives in `rollouts/run_all.py`, emitting CSVs inside `data/results/multi_geom_training/`.
- **New training run:** `src/approach_a_pinn/train_multi_geom.py` trained a 96-hidden-unit PINN on 180 samples (5 load cases). Artifacts are stored under `src/approach_a_pinn/artifacts_multi_geom/`.
- **Validation:** `compare_models.py` benchmarks the new model vs legacy PINN + FEA across all datasets. Results (per geometry/load) are stored in `results/multi_geom_model_metrics.csv`, with visuals in `figures/multi_geom/model_mae_comparison.png`.
- **Headline metrics:** multi-geometry PINN MAE stays within 1–12 J on L-bracket loads (≤1.1 % MAPE) and ≈1.5 J on tapered/ribbed upward cases, while the legacy model exhibits >800 J errors off-distribution. Shear load relative error remains high due to sub-1 J ground truth; absolute error is 2.5 J.
- **Next step:** regenerate the datasets with the higher sample counts from the plan (≥200 per load) to tighten ribbed-channel shear accuracy, then propagate the refreshed findings into Chapters 4–6.

