# Tapered Cantilever Plate Validation Report

_Date:_ 25 Nov 2025  
_Artifacts:_  
- Geometry/load figure – `scenario_validation/figures/pinn/tapered_plate_setup.png`  
- PINN layouts – `scenario_validation/results/pinn_new_part/pinn_tapered_plate_predictions.csv`  
- Differentiable FEA log – `scenario_validation/results/fea_new_part/tapered_plate_diff_fea_log.csv`  
- MMC log – `scenario_validation/results/mmc_new_part/mmc_tapered_plate_log.csv`  
- Metrics – `scenario_validation/results/tapered_plate_pinn_vs_fea_metrics.csv`

---

## 1. Scenario Recap

- **Geometry:** 140 mm × 60 mm tapered plate (thickness tapers from 6 mm at the clamp to 3 mm at the tip) with two screw rows at x = 20 mm and x = 80 mm.
- **Loads/BCs:** Left edge clamp, downward tip force of 450 N, and an additional 25 N-m torsional couple applied at the free end to excite combined bending/torsion.
- **Goal:** Reuse the trained L-bracket PINN, obtain ground-truth FEA via the Julia benchmark, and compare against an MMC optimization run on the same part.

## 2. Methodology

1. **PINN inference:** `pinn_env` was used to run `scenario_validation/pinn/run_tapered_plate_pinn.py`, evaluating 80 sampled layouts (resulting in 9 unique screw combinations due to the discrete rows).
2. **Differentiable FEA:** `scenario_validation/fea/NewPartBenchmark.jl` meshed the tapered plate procedurally and computed compliance for every layout, logging the results alongside elapsed times.
3. **MMC baseline:** `scenario_validation/mmc/run_mmc_tapered_plate.py` (via `mmc_env`) optimized screw locations on a coarse grid with a combined load vector that matches the physical scenario.
4. **Comparison & visuals:** `scenario_validation/compare_tapered_plate.py` merged the datasets, produced figures, and exported summary metrics.

## 3. PINN vs FEA Findings

| Metric | Value |
| --- | --- |
| Samples compared | 762 evaluations (9 unique layouts × repeated logging) |
| FEA compliance range | 22.42 J – 23.57 J |
| PINN prediction range | 0.866 kJ – 1.289 kJ |
| Mean absolute error | **1.11 kJ** |
| Max absolute error | 1.27 kJ |
| Mean relative error | 4.84 × 10³ % |
| 95th percentile relative error | 5.55 × 10³ % |
| Pearson correlation (PINN vs FEA) | 0.13 |

**Diagnosis**

- The PINN was trained solely on the L-bracket geometry; no tapered plates or combined torsion loads were present in the training distribution.
- Feature normalization is dominated by the L-bracket dataset, so the tiny (≈23 J) compliance magnitudes of the tapered plate are outside the learned scale, producing ~1 kJ extrapolations.
- Only two columns of screw positions are allowed in this scenario, further reducing diversity; repeated rows highlight that the model systematically mispredicts every combination rather than occasionally succeeding.

**Implication:** the current surrogate cannot be used for cross-geometry validation. To recover usefulness, the training set must include (at minimum) this tapered plate plus other representative geometries/load cases, followed by retraining/fine-tuning and renewed FEA validation.

## 4. MMC Results

- Best reported compliance: **−6.48 × 10⁵** (solver uses an energy-scaled objective that can become negative as stiffness increases; use relative comparisons rather than absolute units).
- Best layout (iteration 30): screw centers at (21.73 mm, 6.82 mm) and (49.55 mm, 9.00 mm) in the coarse grid coordinates.
- The MMC convergence plot (`scenario_validation/figures/mmc/mmc_convergence_tapered_plate.png`) shows steady improvement until the solver saturates around 30 iterations.

## 5. Recommended Thesis Updates

1. **Dataset expansion:** Add tapered plate samples (and any other benchmarking geometries) to the differentiable FEA rollout pipeline to create a multi-geometry training corpus.
2. **Retraining & normalization:** Recompute statistics and retrain or fine-tune the PINN so it learns the broader compliance range before claiming generalization.
3. **Documentation:** In `project_status_explainer.md` and methodology chapters, explicitly state that the current PINN is limited to the L-bracket distribution and outline the plan to broaden training coverage.
4. **Validation narrative:** Use the plots + metrics above to show how out-of-distribution geometry leads to catastrophic error, motivating the expanded plan.

Once the augmented dataset exists, rerun the same script sequence to demonstrate improved alignment (target MAE ≪ 5 % of the FEA range).

