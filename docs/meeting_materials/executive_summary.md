# Executive Summary: Thesis Progress & Meeting Objectives

**Student:** Diovandi Basheera Putra  
**Supervisor:** Dr. Leonard P. Rusli  
**Date:** 27 November 2025  
**Status:** Technical experiments complete; scope freeze needed

---

## Methodology Briefing

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

---

## Current Results Snapshot

### Multi-Geometry PINN (Approach A)
- **Dataset:** 1020 samples across 3 geometries (L-bracket, tapered plate, ribbed channel)
- **Accuracy:** MAE ≤12.4 J, ≤1.1% MAPE on all high-energy loads
- **Speed:** 0.009 ms/inference (≥10³× faster than FEA at 10.4 ms)
- **Generalization:** One-shot inference across 3 geometries / 5 load cases
- **Training:** 2 hours one-time cost (1500 epochs)

### MMC (Approach B)
- **Geometries:** L-bracket, tapered plate, ribbed channel (all completed)
- **Performance:** 118 ms/iteration, 30 iterations per run
- **Status:** Operational on all three geometries with convergence logs
- **Note:** Compliances in normalized units; conversion to Joules pending

### Comparative Summary
| Method | Time/Iteration | Training Cost | Generalization |
|--------|---------------|---------------|----------------|
| Diff-FEA | 10.4 ms | None | N/A |
| PINN | 0.009 ms | 2 h (one-time) | Cross-geometry |
| MMC | 118 ms | None | Per-load re-optimization |

---

## Proposed Final Tasks Checklist

**Technical Work:**

- [ ] Execute Ansys validation on 3-5 selected layouts (target: <20% error)
- [ ] Normalize MMC compliances to Joules and regenerate unified comparison plots
- [ ] Produce dataset coverage figure for 1020-sample corpus
- [ ] Minor cleanup: fix ribbed-channel shear error if time permits

**Writing Work:**

- [ ] Draft Chapters 1-3 (Introduction, Literature Review, Theoretical Framework)
- [ ] Draft Chapters 4-5 (Methodology, Results)
- [ ] Draft Chapters 6-8 (Comparative Analysis, Discussion, Conclusion)
- [ ] Incorporate supervisor feedback
- [ ] Final formatting and proofreading

---

## Key Decisions Needed

### 1. Scope Freeze
- [ ] **Is current 2D/2.5D work sufficient?** (L-bracket + tapered + ribbed)
- [ ] **Is 3D benchmark required?** If yes:
  - [ ] PINN-only 3D acceptable? (MMC remains 2D-only)
  - [ ] Which geometry: extruded L-bracket or simplified motor-housing?

### 2. Ansys Validation
- [ ] **How many layouts?** (Suggested: 3-5)
- [ ] **Error threshold acceptable?** (Suggested: <20%)
- [ ] **Which layouts?** (Best diff-FEA, best PINN, best MMC per geometry)

### 3. MMC Normalization
- [ ] **Must be done before thesis?** Or acceptable to explain scaling in text?
- [ ] **Timeline:** Before writing or during writing phase?

### 4. Writing Timeline
- [ ] **Chapter delivery schedule:** 
  - Ch. 1-3 draft: _____ weeks
  - Ch. 4-5 draft: _____ weeks
  - Full draft: _____ weeks
- [ ] **Review rounds:** How many? (Suggested: 2-3)
- [ ] **Latest full-draft date:** _____ (back-solve from defense date)

### 5. Thesis Balance
- [ ] **Emphasis:** More on comparative discussion/generalization vs. deep derivations vs. more experiments?
- [ ] **MMC ribbed-channel values:** Acceptable if explained, or must fix first?

---

## Approved Thesis Scope Statement

**One sentence description:**
```









```

**Example options:**

- "Comparative study of multi-geometry PINN vs MMC on three 2D benchmark assemblies with Ansys validation on selected layouts"
- "Comparative study of multi-geometry PINN vs MMC on three 2D benchmark assemblies, with 3D PINN extension, and Ansys validation on selected layouts"

---

## Supervisor Notes

**Decisions made:**

- 
- 
- 

**Additional requirements:**

- 
- 
- 

**Next meeting date:** _____

---

## Key Figures to Reference

- Multi-geometry PINN performance: `data/results/comprehensive_results_table.md`
- Unified speed comparison: `data/results/unified_speed_comparison.png`
- Unified compliance comparison: `data/results/unified_compliance_comparison.png`
- MMC convergence plots: `data/results/mmc_*_compliance.png`
- Full results table: `data/results/comprehensive_results_table.md`

---

**Next Steps:** Fill in decisions during meeting, then freeze scope and proceed with final tasks.

