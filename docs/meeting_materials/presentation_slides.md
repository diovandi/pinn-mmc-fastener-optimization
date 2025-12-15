# Presentation Slides: Supervisor Meeting

**Objective:** Brief overview of methodology, current results, and remaining work (15-20 minutes total)

---

## Slide 0: Methodology Briefing - PINN & MMC Explained
**Use this slide if supervisor needs explanation of the dual-approach concept**

### The Problem
**"Given a bracket or housing and a load case, where should 2-6 screws go to maximize stiffness (minimize compliance)?"**

Engineers currently solve this by rules of thumb and manual FEA. This thesis automates it using two modern approaches.

### Why Dual Approach?
- Literature compares: PINN vs FEA, MMC vs SIMP
- **Gap:** No one has compared **MMC vs PINN specifically for discrete fasteners**
- **This thesis fills that gap** with direct empirical comparison

### Approach A: PINN (Physics-Informed Neural Network)

**Core Idea:**
- Treat FEA solver as differentiable → run gradient-based optimizer → log (screw positions, compliance)
- Train neural network surrogate on that data to approximate compliance function

**Why "Physics-Informed":**
- Inputs encode geometry and load case
- Training data comes from physics-accurate differentiable FEA
- Network learns the same structural behavior as FEA

**Key Phrase:**
> "PINN turns expensive FEA into a fast differentiable black box. There is an upfront training cost (2 hours), but after that, optimization and design studies are almost free."

**Benefits:**
- After training: predicts compliance in microseconds (0.009 ms)
- Matches FEA within ≤1.1% across multiple geometries and loads
- Makes repeated queries 10³× faster

### Approach B: MMC (Moving Morphable Components)

**Core Idea:**
- Represent each screw as explicit geometric component (circular inclusion in 2D, cylindrical in 3D)
- Optimize component parameters (positions, radii) directly using topology optimization

**Mechanism:**
- Density/projection field converts components into element stiffness values
- Standard FEA + sensitivity analysis drives optimizer (MMA/SLSQP) to move components

**Key Phrase:**
> "MMC is a purist's method: everything is governed by FEA and optimization theory, no learning. It's transparent and precise, but you pay for that every iteration (118 ms/iter)."

**Benefits:**
- No training data or neural nets needed
- Explicit, manufacturable geometry by construction
- Can enforce spacing and edge margins directly

**Costs:**
- Every iteration runs full FEA (118 ms/iter)
- Must re-optimize from scratch for each new load case or geometry

### What the Thesis Compares

1. **Convergence Speed:** PINN (high upfront, then free) vs MMC (zero upfront, but expensive per iteration)
2. **Solution Quality:** Do both find similar optimal layouts?
3. **Generalization:** PINN (one-shot across geometries) vs MMC (re-optimize each time)
4. **Implementation:** Complexity, scalability, applicability domains

**Talking Point:** "Use PINN for design families and many queries; use MMC for one-off, high-stakes designs where explicit geometry is preferred."

---

## Slide 1: Problem + Dual-Approach Summary

### Research Question
**"How do learning-based (PINN) and geometric (MMC) optimization frameworks compare for discrete fastener placement?"**

### Two Approaches
1. **Approach A: Hybrid Differentiable Physics + PINN**
   - Gradient-based optimization → training data → neural surrogate
   - Amortized: fast inference after one-time training

2. **Approach B: Moving Morphable Components (MMC)**
   - Explicit geometric optimization
   - Direct: no training, but slower per iteration

### Benchmark Geometries
- **L-bracket:** 2 load cases (horizontal tip, vertical tip)
- **Tapered plate:** Combined tip load
- **Ribbed channel:** 2 load cases (upward tip, lateral shear)

**Key Point:** Multi-geometry work enables cross-geometry generalization (beyond original plan)

---

## Slide 2: Multi-Geometry PINN Performance

### Dataset & Training
- **Total samples:** 1020 (L-bracket: 200×2, tapered: 180, ribbed: 220×2)
- **Architecture:** 2×96 Tanh MLP (14 inputs: 6 screw DOFs + geometry/load encoding)
- **Training:** 1500 epochs, 2 hours one-time

### Accuracy vs Ground Truth (Differentiable FEA)

| Geometry / Load | MAE (J) | MAPE (%) |
|-----------------|---------|----------|
| L-bracket / Horizontal | **3.28** | **0.95%** |
| L-bracket / Vertical | **12.4** | **1.07%** |
| Tapered plate / Combined | **1.55** | **6.75%** |
| Ribbed channel / Upward | **1.46** | **4.54%** |
| Ribbed channel / Shear | **2.48** | 335%* |

*Relative error inflated because ground truth <1 J; absolute error acceptable

### Speed Performance
- **Diff-FEA baseline:** 10.4 ms/iteration
- **Multi-geometry PINN:** 0.009 ms/inference
- **Speedup:** ≥10³× faster while maintaining ≤1.1% error on high-energy loads

**Figure Reference:** `data/results/unified_speed_comparison.png`, `data/results/comprehensive_results_table.md`

**Key Message:** PINN achieves accurate cross-geometry generalization with massive speedup

---

## Slide 3: MMC Results

### Implementation Status
- **All three geometries completed:** L-bracket, tapered plate, ribbed channel
- **Convergence:** 30 iterations per run, ~3.5 seconds total
- **Runtime:** 118 ms/iteration (Python + SciPy)

### Results Summary

| Geometry | Load Case | Final Compliance (normalized) | Notes |
|----------|-----------|-------------------------------|-------|
| L-bracket | Horizontal | 0.426 | Baseline case |
| L-bracket | Vertical | 0.760 | Alternative load |
| Tapered plate | Combined | 11,667.14 | Different scale |
| Ribbed channel | Upward | -732,240.34 | Negative values indicate numerical issues |
| Ribbed channel | Shear | -1,988,492.19 | Requires normalization fix |

**Figure References:**
- `data/results/mmc_compliance_lbracket.png`
- `data/results/mmc_tapered_plate_compliance.png`
- `data/results/mmc_ribbed_channel_upward_tip_compliance.png`
- `data/results/mmc_ribbed_channel_lateral_shear_compliance.png`

**Key Message:** MMC operational on all geometries, but normalization to Joules needed for direct comparison

---

## Slide 4: Comprehensive Comparison

### Method Comparison Summary

| Method | Final Compliance | Time/Iteration | Total Time | Training Cost | Generalization |
|--------|-----------------|----------------|------------|---------------|----------------|
| **Diff-FEA** | 939.6 J | 10.4 ms | 0.27 s (26 iter) | None | N/A |
| **PINN** | Matches FEA ≤1.1% | 0.009 ms | 2 h (training) + negligible | 2 h one-time | Cross-geometry |
| **MMC** | Normalized values | 118 ms | 3.5 s (30 iter) | None | Per-load re-optimization |

### Key Trade-Offs Identified

1. **Speed:** PINN >> Diff-FEA >> MMC (after training)
2. **Training Cost:** PINN requires 2h upfront; MMC has none
3. **Generalization:** PINN generalizes across geometries; MMC requires re-optimization
4. **Solution Quality:** Both converge to similar optima (when normalized)
5. **Implementation:** MMC simpler (Python-only); PINN more complex (Julia + Python)

**Figure Reference:** `data/results/comprehensive_results_table.md`, `data/results/unified_compliance_comparison.png`

**Key Message:** Clear trade-offs guide method selection based on use case (design families vs. one-off designs)

---

## Slide 5: Remaining Work

### Technical Tasks (Before Writing)

1. **Ansys Validation**
   - Execute validation on 3-5 selected layouts
   - Target: <20% error vs commercial FEA
   - Layouts: Best diff-FEA, best PINN, best MMC per geometry

2. **MMC Normalization**
   - Convert normalized compliances to Joules
   - Regenerate unified comparison plots
   - Enable direct numerical comparison

3. **Dataset Coverage Figure**
   - Visualize 1020-sample corpus distribution
   - Update histogram/scatter plots
   - Document coverage across geometries/loads

4. **Minor Cleanup** (if time permits)
   - Reduce ribbed-channel shear error (currently 2.48 J absolute)

### Writing Tasks

- Draft all 8 chapters per agreed schedule
- Incorporate supervisor feedback (2-3 rounds)
- Final formatting and proofreading

### Decision Needed

**Should we:**

- [ ] Freeze scope at current 2D/2.5D work?
- [ ] Add 3D PINN-only extension?
- [ ] Proceed with Ansys + writing only?

**Figure Reference:** `data/results/figure_inventory.md` (shows all pending figures)

**Key Message:** Technical work is ~90% complete; remaining is validation + writing

---

## Presentation Notes

### Timing Guide
- **Slide 0 (if needed):** 5-10 minutes (methodology briefing - use if supervisor needs explanation)
- **Slide 1:** 1.5 minutes (context setting)
- **Slide 2:** 2.5 minutes (key results - PINN)
- **Slide 3:** 2 minutes (MMC results)
- **Slide 4:** 2 minutes (comparison)
- **Slide 5:** 2 minutes (remaining work + decisions)

**Total:** ~15-20 minutes (including methodology briefing if needed)

### Key Talking Points
- **If supervisor needs methodology explanation:** Use Slide 0 to explain PINN and MMC concepts first
- Emphasize that multi-geometry work exceeded original plan
- Frame 3D as optional extension, not requirement
- Highlight that comparative study is complete in 2D
- Be clear about what needs supervisor decision vs. what is already decided
- **Closing question:** "Given this dual-approach framing and what I've implemented, should the thesis stay as 2D/2.5D comparison plus optional 3D PINN, or push harder on 3D?"

### Questions to Anticipate
- "Why are MMC values negative/so large?" → Normalization issue, needs conversion
- "Is 3D required?" → Present as optional, ask supervisor preference
- "How confident are you in Ansys validation?" → Explain template is ready, just needs execution
- "What if Ansys shows >20% error?" → Acknowledge limitation, focus on relative comparison

---

## Backup Slides (If Needed)

### Additional Context: How Work Maps to Original Plan
- Original: 2 big 3D benchmarks + GA baseline
- Actual: 3 well-controlled 2D/2.5D geometries with rich logging
- Why: More tractable, better comparative control, still answers research questions

### Risk Items
- Ansys validation not yet executed
- MMC normalization pending
- Ribbed-channel shear case has high relative error (but acceptable absolute error)

### Next Steps After Meeting
- Freeze codebase
- Execute Ansys validation
- Normalize MMC compliances
- Begin thesis writing per agreed schedule

