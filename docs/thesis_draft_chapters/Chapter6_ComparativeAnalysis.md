# Chapter 6: Comparative Analysis

## 6.1 Convergence Speed Comparison

### 6.1.1 Per-Iteration Performance

The refreshed unified speed comparison (`unified_speed_comparison.png`) demonstrates:

- **Diff-FEA (Julia):** 10.4 ms/iter — baseline differentiable solver.
- **Multi-geometry PINN (PyTorch):** ~0.009 ms/sample — >1 000× faster even after widening the network for 14 input features.
- **Legacy PINN:** 0.047 ms/sample — retained for ablations; still ≈222× faster than diff-FEA.
- **MMC (Python):** 118 ms/iter — dominated by full FEA solve per iteration.

**Key Insight:** The new dataset doubles the feature space but barely dents runtime; the one-time training cost (~2 h for 1 500 epochs) amortizes across all geometries/loads, whereas MMC has no training cost but must re-run FEA per design/load.

### 6.1.2 Total Optimization Time

- **Multi-geometry PINN:** ~2 h training (1 500 epochs) + negligible inference per design (milliseconds for 26 iterations).
- **Legacy PINN:** ~2–4 h training depending on batch count; inference negligible.
- **MMC:** 30 iterations × 118 ms ≈ 3.5 s per run (per load case).
- **Diff-FEA:** 26 iterations × 10.4 ms ≈ 0.27 s per run.

**Trade-off:** MMC remains the quickest for a single one-off optimization, but the multi-geometry PINN amortizes its cost the moment multiple geometries/loads are queried. The refreshed dataset means one checkpoint now serves three geometries and five loads without retraining.

## 6.2 Solution Quality Comparison

### 6.2.1 Final Compliance Values

- **Diff-FEA / Multi-geometry PINN:** 939.6 J on the L-bracket benchmark (PINN predictions track diff-FEA within ≤1 % MAPE on both loads).
- **MMC:** 0.426 (normalized) for horizontal, 0.760 for vertical. Conversion to Joules requires the shared post-processing scripts; this is captured in `method_comparison.csv`.
- **Other geometries:** PINN predictions on tapered plate (≈23 J) and ribbed channel upward (~32 J) align with differentiable FEA within 1–7 % relative error; MMC runs for these geometries are queued next.

### 6.2.2 Spatial Distribution

- L-bracket: Both methods bias screws toward the high-strain inner corner; MMC exposes migration trajectories, while PINN recovers the pattern from data.
- Tapered plate / Ribbed channel: PINN inference now provides immediate screw placement guidance for these geometries; MMC results will add interpretable counterparts once the new runs complete.

## 6.3 Generalization Capability

### 6.3.1 PINN Generalization

- With the multi-geometry dataset, PINN generalization is now **quantitative** rather than hypothetical: ≤1.1 % MAPE on both L-bracket loads, 6.75 % on the tapered plate combined load, 4.54 % on ribbed channel upward, and 2.48 J absolute on ribbed channel shear (due to sub-1 J targets).
- Remaining work: scale the ribbed-channel shear dataset and add force-weighting so absolute error drops below 1 J.

### 6.3.2 MMC Re-optimization

- MMC still requires a full optimization run per geometry/load, but its constraint transparency remains valuable. Extending MMC to tapered/ribbed will supply direct baselines for the new PINN results.

**Trade-off:** Multi-geometry PINN now provides instant cross-geometry inference with documented accuracy, while MMC guarantees physics fidelity per run but incurs re-optimization cost.

## 6.4 Implementation Complexity

### 6.4.1 Code Complexity

- **PINN:** ~800 LOC (Julia + Python), two-language implementation, AD debugging challenges
- **MMC:** ~400 LOC (Python only), straightforward FEA integration

### 6.4.2 Dependencies

- **PINN:** Julia, Zygote.jl, PyTorch, CUDA (optional)
- **MMC:** NumPy, SciPy, Matplotlib

**Insight:** MMC is simpler to implement and debug, while PINN requires expertise in both automatic differentiation and neural network training.

## 6.5 Applicability Domains

### 6.5.1 When to Use PINN

- Design families requiring many optimizations
- Fast iteration during design exploration
- Generalization to similar load cases is valuable

### 6.5.2 When to Use MMC

- Single design optimization
- Need for explicit geometric control
- Interpretable component trajectories
- No training data available

## 6.6 Summary Table

| Criterion | Multi-Geometry PINN | MMC | Winner |
|----------|---------------------|-----|--------|
| Speed (after training) | 0.009 ms/sample | 118 ms/iter | PINN |
| Training Cost | ~2 h (1 500 epochs) | None | MMC |
| Solution Quality (L-bracket) | Matches diff-FEA within ≤1 % | 0.426 (norm) | Tie* |
| Cross-Geometry Accuracy | ≤6.75 % MAPE (tapered/ribbed upward) | Pending runs | PINN (current) |
| Implementation Complexity | Julia + Python + PyTorch | Python + SciPy | MMC |
| Interpretability | Learned surrogate (low) | Explicit component motion (high) | MMC |

*Requires converting MMC normalized compliance to Joules for strict equality.

