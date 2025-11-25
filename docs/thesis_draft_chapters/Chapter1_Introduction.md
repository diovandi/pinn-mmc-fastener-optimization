# Chapter 1: Introduction

## 1.1 Background and Motivation

Mechanical assemblies rely on discrete fasteners—screws, bolts, and rivets—to transfer loads, remove degrees of freedom between rigid bodies, and enable assembly and disassembly. The placement of these fasteners critically affects structural performance, yet current engineering practice relies heavily on manual design using heuristics (e.g., "place screws every 50mm"), finite element analysis (FEA) validation after-the-fact, and iterative trial-and-error with physical prototypes.

**The Problem:** No autonomous framework exists that optimizes discrete fastener positions for structural performance while completing optimization in minutes-to-hours rather than weeks, and remains tractable for practical implementation.

## 1.2 Research Objectives

This thesis presents a **first-of-its-kind comparative study** between two fundamentally different optimization paradigms for discrete fastener placement:

1. **Approach A: Hybrid Differentiable Physics + Physics-Informed Neural Network (PINN)** — A learning-based framework that combines gradient-based optimization through differentiable FEA with neural network surrogate acceleration
2. **Approach B: Moving Morphable Components (MMC)** — An explicit geometric optimization framework that parameterizes each fastener as a morphable component and optimizes directly through FEA

Both methods target the same objective: **minimizing structural compliance (maximizing stiffness)** by autonomously optimizing discrete screw/bolt positions on mechanical assemblies.

## 1.3 Research Questions

**Primary Question:** How do learning-based (Differentiable Physics + PINN) and geometric (MMC) optimization frameworks compare in terms of convergence speed, solution quality, generalization capability, and implementation feasibility for discrete fastener placement in mechanical assemblies?

**Sub-Questions:**
- **RQ1:** Does the PINN surrogate achieve faster optimization convergence than direct MMC optimization?
- **RQ2:** Do both methods converge to similar optimal screw placements and compliance values?
- **RQ3:** When trained on one load case, can PINN generalize to unseen load cases? Does MMC require re-optimization?
- **RQ4:** Which method is more tractable for implementation? What are the debugging challenges and computational requirements?
- **RQ5:** For what scenarios (one-off design vs. design families, simple vs. complex geometries) is each method preferred?

## 1.4 Thesis Structure

- **Chapter 2:** Literature Review — Differentiable physics, PINNs, and MMC topology optimization
- **Chapter 3:** Theoretical Framework — Mathematical foundations of both approaches
- **Chapter 4:** Methodology — Implementation details and benchmark setup
- **Chapter 5:** Results — Performance analysis of both methods on L-bracket benchmark
- **Chapter 6:** Comparative Analysis — Direct comparison of speed, quality, and generalizability
- **Chapter 7:** Discussion — Trade-offs, limitations, and applicability domains
- **Chapter 8:** Conclusion — Summary, contributions, and future work

## 1.5 Preview of Results

The refreshed multi-geometry study demonstrates:

- **PINN Surrogate:** A single 2×96 Tanh MLP (trained on 180 differentiable-FEA samples across the L-bracket, tapered plate, and ribbed channel) achieves ≤1.1 % MAPE on all high-energy loads, 6.75 % on the tapered plate combined load, and 4.54 % on ribbed channel upward, while inferring at ~0.009 ms/sample (>1 000× faster than diff-FEA).
- **MMC Framework:** Converges within ~30 iterations on the L-bracket benchmark with explicit spacing/edge constraints; infrastructure is ready to extend to the new geometries for full comparative coverage.
- **Dataset Pipeline:** Scenario-validation rollouts now produce balanced multi-geometry datasets (`data/results/multi_geom_training/`), with a roadmap to ≥200 samples per load to reduce the remaining ribbed-channel shear error below 1 J.

These results provide the quantitative foundation for the comparative analysis presented in subsequent chapters.

