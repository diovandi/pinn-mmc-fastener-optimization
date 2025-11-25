# Comparative Framework for Optimal Discrete Fastener Placement: Hybrid Differentiable Physics + PINN vs. Moving Morphable Components (MMC)

## Complete Bachelor Thesis Plan — Dual-Method Comparative Study

**Title:** Comparative Evaluation of Geometric and Learning-Based Optimization Frameworks for Discrete Fastener Placement: A Hybrid Differentiable Physics + PINN versus Moving Morphable Components Study

**Subtitle:** AI-Driven and Explicit Geometric Approaches to Generative Constraint Synthesis in Mechanical Assemblies

**Student:** Diovandi Basheera Putra  
**Institution:** Swiss German University  
**Department:** Mechanical Engineering (Mechatronics Focus)  
**Duration:** 8 Months (32 Weeks)  
**Supervisor:** Leonard P. Rusli, B.Sc, M.Sc, Ph.D  
**Thesis Type:** Bachelor (Reduced-scope, degree-sufficient, comparative methodology study)

---

## Executive Summary

This thesis presents a **first-of-its-kind comparative study** between two fundamentally different optimization paradigms for discrete fastener placement in mechanical assemblies:

1. **Approach A: Hybrid Differentiable Physics + Physics-Informed Neural Network (PINN)** — A learning-based framework that combines gradient-based optimization through differentiable FEA with neural network surrogate acceleration
2. **Approach B: Moving Morphable Components (MMC)** — An explicit geometric optimization framework that parameterizes each fastener as a morphable component and optimizes directly through FEA

Both methods target the same objective: **minimizing structural compliance (maximizing stiffness)** by autonomously optimizing discrete screw/bolt positions on 3D mechanical assemblies. This comparative study reveals the **trade-offs, strengths, weaknesses, and applicability domains** of geometric versus learning-based methods for discrete placement problems.

### Key Research Contribution

**No existing literature directly compares geometric topology optimization (MMC) with physics-informed machine learning (PINN) for discrete mechanical fastener placement.** This thesis fills that gap and provides:

- First implementation of **both paradigms** on identical benchmark assemblies
- Quantitative comparison of **convergence speed, solution quality, generalizability, and implementation complexity**
- Practical guidance for engineers choosing between geometric and learning-based frameworks
- Open-source implementations of both methods for reproducibility

### Novelty Claims

1. **Primary Novelty:** First direct comparison of MMC vs. PINN for discrete fastener optimization
2. **Secondary Novelty (Approach A):** Active learning paradigm where optimizer generates PINN training data
3. **Secondary Novelty (Approach B):** First adaptation of MMC framework to discrete fastener synthesis
4. **Methodological Novelty:** Unified validation protocol for hybrid learning-based and explicit geometric methods

### Execution Status (Nov 25 2025)
- Multi-geometry PINN surrogate trained on 180 differentiable-FEA samples (L-bracket horizontal/vertical, tapered plate combined, ribbed channel upward/shear) with MAE ≤12.4 J (≤1.1 % MAPE) on high-energy loads and 1.55 J (6.75 %) on the tapered plate.
- Surrogate inference remains ~0.009 ms/sample, maintaining ≥1 000× acceleration over the 10.4 ms/iter differentiable FEA baseline and ≈13 000× over MMC (118 ms/iter).
- MMC implementation is operational on the L-bracket (horizontal + vertical loads) and ready for tapered plate / ribbed channel extensions using the existing domain/constraint configs.
- Outstanding technical work: scale each load case to ≥200 samples per `dataset_plan.yaml`, run MMC on the additional geometries, and execute Ansys/CalculiX validation on the refreshed layouts.

---

## 1. Background & Motivation

### 1.1. The Discrete Placement Problem in Mechanical Design

Mechanical assemblies rely on discrete fasteners (screws, bolts, rivets) to:

- Remove degrees of freedom between rigid bodies
- Transfer loads across interfaces
- Enable assembly/disassembly

**Current State:** Engineers place fasteners manually using:

- Heuristics (e.g., "place screws every 50mm")
- Finite Element Analysis (FEA) validation after-the-fact
- Trial-and-error with physical prototypes

**The Gap:** No autonomous framework exists that:

- Optimizes discrete fastener positions for structural performance
- Completes optimization in minutes-to-hours (not weeks)
- Is tractable for bachelor-level research implementation

### 1.2. Two Paradigms for Discrete Optimization

Recent advances (2020-2025) in computational mechanics offer two fundamentally different solutions:

#### Paradigm 1: Learning-Based (Differentiable Physics + PINN)
- **Philosophy:** Learn a surrogate model of the FEA physics from optimization trajectories
- **Mechanism:** Gradient-based optimization through differentiable FEA → collect samples → train PINN → accelerate future optimizations
- **Strengths:** Ultra-fast inference (~1ms), learns generalizable patterns, active learning
- **Challenges:** Requires training data, neural network debugging, two-language implementation

#### Paradigm 2: Geometric (Moving Morphable Components)
- **Philosophy:** Parameterize each fastener as an explicit geometric component with continuous variables
- **Mechanism:** Direct optimization of geometric parameters through FEA in reduced-dimension design space
- **Strengths:** No training data needed, mathematically rigorous, crisp boundaries, interpretable
- **Challenges:** Direct FEA cost per iteration, fixed topology (no component addition/removal)

### 1.3. Research Positioning

This thesis does **not** attempt to prove one method is universally superior. Instead, it provides:

- **Empirical evidence** of performance on standardized benchmarks
- **Trade-off analysis** for different use cases (single design vs. design families)
- **Implementation insights** for practitioners

---

## 2. Research Problem Statement

### 2.1. The Core Question

**"How do learning-based (Differentiable Physics + PINN) and geometric (MMC) optimization frameworks compare in terms of convergence speed, solution quality, generalization capability, and implementation feasibility for discrete fastener placement in mechanical assemblies?"**

### 2.2. Sub-Research Questions

**RQ1: Convergence Speed**  
Does the PINN surrogate achieve faster optimization convergence than direct MMC optimization? What is the trade-off between one-time PINN training cost vs. per-iteration MMC cost?

**RQ2: Solution Quality**  
Do both methods converge to similar optimal screw placements and compliance values? If different, which produces more manufacturable layouts?

**RQ3: Generalization**  
When trained on one load case, can PINN generalize to unseen load cases? Does MMC require re-optimization for each new load case?

**RQ4: Implementation Feasibility**  
Which method is more tractable for bachelor-level implementation? What are the debugging challenges, computational requirements, and software dependencies?

**RQ5: Applicability Domains**  
For what scenarios (one-off design vs. design families, simple vs. complex geometries, 2D vs. 3D) is each method preferred?

### 2.3. Hypothesis

**H1:** PINN achieves 10-100× faster inference after training but requires 2-4 hours initial training cost  
**H2:** Both methods converge to similar compliance values (within 10%) on identical benchmarks  
**H3:** MMC produces more regular, manufacturable layouts due to explicit geometric constraints  
**H4:** PINN generalizes to unseen load cases with 10-20% error; MMC requires full re-optimization  
**H5:** MMC is simpler to implement (Python-only, existing code) vs. PINN (Julia + Python, AD + neural network training)

---

## 3. Goals & Objectives

### 3.1. Primary Goal

Develop and validate **both** optimization frameworks on identical benchmark assemblies, producing a **comprehensive comparative analysis** that guides future research and industrial application of discrete placement optimization.

### 3.2. Specific Objectives

**O1: Implement Approach A (Differentiable Physics + PINN)**

- Build differentiable static FEA solver in Julia + Zygote.jl
- Implement gradient-based screw placement optimizer (ADAM)
- Generate 300-500 training samples from optimization runs
- Train PyTorch PINN to predict compliance from screw configurations
- Achieve <10% prediction error vs. ground truth FEA

**O2: Implement Approach B (MMC Framework)**

- Port/adapt MMC topology optimization code (Python + FEniCS)
- Parameterize fasteners as circular/cylindrical geometric components
- Implement geometric constraints (spacing, edge margins)
- Optimize using Method of Moving Asymptotes (MMA) or SLSQP
- Converge to discrete screw placements in <30 minutes per run

**O3: Establish Unified Validation Protocol**

- Select 2 benchmark assemblies (L-bracket, motor housing cover)
- Define common metrics: compliance, convergence time, screw coordinates
- Export optimized layouts to Ansys for high-fidelity validation
- Compare vs. baseline Genetic Algorithm + Full FEA

**O4: Conduct Comprehensive Comparative Analysis**

- Generate convergence plots (compliance vs. iteration/time)
- Compare final screw placements and compliance values
- Analyze generalization to Load Case 2 (train on Case 1)
- Document implementation complexity, debugging time, LOC

**O5: Deliver Open-Source Implementations**

- Publish Julia DiffFEA module + PINN training scripts
- Publish Python MMC fastener optimizer
- Provide benchmark CAD files and validation protocols
- Write comprehensive documentation and tutorials

---

## 4. Methodology: Dual-Framework Architecture

### 4.1. System Overview

```
                    ┌──────────────────────────────────────┐
                    │   INPUT: CAD Assembly (STEP file)    │
                    │   Load cases, BCs, material props    │
                    └──────────────────────────────────────┘
                                     ↓
                    ┌──────────────────────────────────────┐
                    │   CAD Processing Pipeline (SHARED)   │
                    │   • Voxelization: 64³ grid           │
                    │   • Surface mesh extraction          │
                    │   • Load/BC channel encoding         │
                    └──────────────────────────────────────┘
                         ↙                           ↘
        ┌─────────────────────────┐      ┌─────────────────────────┐
        │   APPROACH A: PINN      │      │   APPROACH B: MMC       │
        └─────────────────────────┘      └─────────────────────────┘
                 ↓                                    ↓
    ┌──────────────────────────┐       ┌──────────────────────────┐
    │ Phase 1: Diff FEA        │       │ Geometric Optimization   │
    │ • Julia + Zygote.jl      │       │ • Python + FEniCS        │
    │ • ADAM optimizer         │       │ • MMA/SLSQP optimizer    │
    │ • Log 300-500 samples    │       │ • Direct FEA per iter    │
    │ • Time: ~10-30min        │       │ • Time: ~20-40min        │
    └──────────────────────────┘       └──────────────────────────┘
                 ↓                                    ↓
    ┌──────────────────────────┐       ┌──────────────────────────┐
    │ Phase 2: PINN Training   │       │ Constraint Projection    │
    │ • PyTorch 3D CNN + MLP   │       │ • Min spacing: 20mm      │
    │ • Physics loss included  │       │ • Edge margin: 10mm      │
    │ • Train time: 2-4 hours  │       │ • Surface projection     │
    │ • Inference: ~1ms        │       │ • Volume fraction        │
    └──────────────────────────┘       └──────────────────────────┘
                 ↓                                    ↓
    ┌──────────────────────────┐       ┌──────────────────────────┐
    │ Phase 3: PINN-Accelerated│       │ Output: Screw Positions  │
    │ • Replace FEA with PINN  │       │ • [x1,y1,z1,...,xn,yn,zn]│
    │ • 10-100× speedup        │       │ • Material density field │
    │ • Test generalization    │       │ • Compliance value       │
    └──────────────────────────┘       └──────────────────────────┘
                 ↓                                    ↓
                    ┌──────────────────────────────────────┐
                    │   UNIFIED VALIDATION (SHARED)        │
                    │   • Export to Ansys for validation   │
                    │   • Compare compliance predictions   │
                    │   • Visualize screw placements (3D)  │
                    │   • Generate comparison tables/plots │
                    └──────────────────────────────────────┘
                                     ↓
                    ┌──────────────────────────────────────┐
                    │   COMPARATIVE ANALYSIS               │
                    │   • Convergence speed comparison     │
                    │   • Solution quality metrics         │
                    │   • Generalization error analysis    │
                    │   • Implementation complexity review │
                    └──────────────────────────────────────┘
```

### 4.2. Approach A: Hybrid Differentiable Physics + PINN

#### 4.2.1. Ground Truth Physics: Differentiable Static FEA

**Implementation:**

- **Language:** Julia (for performance and automatic differentiation)
- **AD Library:** Zygote.jl (reverse-mode automatic differentiation)
- **Element Type:** Linear tetrahedral elements (C3D4)
- **Solver:** Direct sparse LU factorization
- **Objective:** Compliance$C = \frac{1}{2} \mathbf{u}^T \mathbf{K} \mathbf{u}$

**Differentiable Forward Pass:**
```julia
function compliance_loss(screw_positions::Vector{Float64}, geometry, loads, BCs)
    # 1. Project screw positions onto voxel grid with Gaussian kernels
    material_field = voxel_grid + gaussian_screws(screw_positions, σ=2)
    
    # 2. Assemble stiffness matrix K(material_field)
    K = assemble_stiffness(material_field, E_base, E_screw)
    
    # 3. Apply boundary conditions
    K_bc, f_bc = apply_BCs(K, loads, BCs)
    
    # 4. Solve for displacement: K*u = f
    u = K_bc \ f_bc
    
    # 5. Compute compliance
    C = 0.5 * dot(u, K_bc * u)
    
    return C
end

# Gradient via automatic differentiation
∇C = Zygote.gradient(compliance_loss, screw_positions)[1]
```

**Key Innovation:** The gradient ∇C with respect to continuous screw coordinates (x, y, z) is computed automatically, enabling gradient-based optimization of discrete placements.

#### 4.2.2. Phase 1: Gradient-Based Screw Placement

**Optimizer:** ADAM (adaptive learning rate)  

**Design Variables:** $\mathbf{p} = [x_1, y_1, z_1, \ldots, x_n, y_n, z_n] \in \mathbb{R}^{3n}$ 

**Constraints:**

- Minimum spacing:$\|\mathbf{p}_i - \mathbf{p}_j\| \geq 20 \text{mm}$
- Surface projection: Screws must lie on part surfaces
- Bounding box:$x_{\text{min}} \leq x_i \leq x_{\text{max}}$

**Algorithm:**
```
Initialize: p = [random positions on surface]
for iteration in 1:300:
    # Forward pass: Compute compliance and gradient
    C = compliance_loss(p, geometry, loads, BCs)
    ∇C = Zygote.gradient(compliance_loss, p)
    
    # ADAM update
    p = ADAM_update(p, ∇C, learning_rate=0.01)
    
    # Project constraints
    p = enforce_min_spacing(p, d_min=20mm)
    p = project_to_surface(p, mesh)
    
    # Log data every 5 iterations
    if mod(iteration, 5) == 0:
        save_sample(p, C, ∇C)
```

**Output:** 

- Optimized screw positions
- Training dataset: 300-500 samples of (geometry, p, C)
- Convergence time: ~10-30 minutes

#### 4.2.3. Phase 2: PINN Surrogate Training

**Architecture:**
```
Input: Voxel grid (64×64×64×4 channels)
       Channels: [geometry, screw Gaussians, BCs, loads]
       
Encoder: 3D CNN
       Layer 1: Conv3D(4 → 32 filters, kernel=3, stride=2) + ReLU
       Layer 2: Conv3D(32 → 64 filters, kernel=3, stride=2) + ReLU
       Layer 3: Conv3D(64 → 128 filters, kernel=3, stride=2) + ReLU
       
Flatten: 128 × 8 × 8 × 8 = 65536 → 512
       
Latent MLP:
       Dense(512 → 256) + SiLU
       Dense(256 → 128) + SiLU
       
Output: Dense(128 → 1) = Compliance prediction
```

**Loss Function:**
$$
\mathcal{L} = \text{MSE}(C_{\text{PINN}}, C_{\text{FEA}}) + \lambda_1 \|\mathbf{K}(\mathbf{p})\mathbf{u} - \mathbf{f}\|^2 + \lambda_2 \|\text{BC}_{\text{residual}}\|^2
$$

- **Data loss:** Mean squared error between PINN prediction and FEA ground truth
- **Physics loss (equilibrium):** Residual of static equilibrium equation
- **Boundary loss:** Satisfaction of boundary conditions

**Training:**

- Framework: PyTorch 2.0+
- Optimizer: Adam (lr=1e-4)
- Batch size: 16
- Epochs: ~500-1000 (early stopping based on validation loss)
- Hardware: 1× NVIDIA RTX 3060 (12GB VRAM)
- Training time: 2-4 hours

**Validation:**

- 80/20 train-validation split
- Target: MAE < 10% on validation set

#### 4.2.4. Phase 3: PINN-Accelerated Optimization

Replace the FEA call in the optimization loop with PINN inference:

```julia
for iteration in 1:100:
    # Ultra-fast forward pass via PINN
    C = PINN_predict(p, geometry, loads)  # ~1ms
    ∇C = PyTorch.autograd.grad(C, p)      # ~1ms
    
    p = ADAM_update(p, ∇C)
    p = enforce_constraints(p)
```

**Expected speedup:** 10-100× faster per iteration compared to full FEA

### 4.3. Approach B: Moving Morphable Components (MMC)

#### 4.3.1. Geometric Parameterization

Each fastener is represented as a **geometric component** with explicit parameters:

**2D Fastener:**

- Center:$(x_i, y_i)$
- Radius:$r_i = 5 \text{mm}$(fixed)
- Material:$E_{\text{bolt}} = 200 \text{GPa}$

**3D Fastener:**

- Center:$(x_i, y_i, z_i)$
- Radius:$r_i = 5 \text{mm}$
- Orientation vector:$\hat{\mathbf{n}}_i$(perpendicular to surface)

**Design Vector:**
$$
\mathbf{X} = [x_1, y_1, z_1, \ldots, x_n, y_n, z_n] \in \mathbb{R}^{3n}
$$

#### 4.3.2. Material Mapping (Ersatz Model)

MMC uses a **level-set-like projection** to assign material properties:

For each finite element$e$at centroid$\mathbf{x}_e$:
$$
\rho_e(\mathbf{X}) = \max_{i=1,\ldots,n} H\left( r_i - \|\mathbf{x}_e - \mathbf{c}_i\| \right)
$$

where$H(\cdot)$is a smoothed Heaviside function.

**Material interpolation:**
$$
E_e = E_{\text{min}} + \rho_e (E_{\text{bolt}} - E_{\text{min}})
$$

where$E_{\text{min}} = 10^{-6} E_{\text{bolt}}$(void stiffness to avoid singularity).

#### 4.3.3. FEA and Sensitivity Analysis

**Static equilibrium:**
$$
\mathbf{K}(\mathbf{X}) \mathbf{u} = \mathbf{f}
$$

**Objective function (compliance minimization):**
$$
\min_{\mathbf{X}} C(\mathbf{X}) = \frac{1}{2} \mathbf{u}^T \mathbf{K}(\mathbf{X}) \mathbf{u}
$$

**Sensitivity (adjoint method):**
$$
\frac{\partial C}{\partial x_i} = -\mathbf{u}^T \frac{\partial \mathbf{K}}{\partial x_i} \mathbf{u}
$$

**Implementation:**

- FEA solver: FEniCS (Python)
- Element type: Triangular (2D) or tetrahedral (3D)
- Mesh: Fixed background mesh (64×64 or 128×128)

#### 4.3.4. Optimization Algorithm

**Optimizer:** Method of Moving Asymptotes (MMA) or SLSQP

**Constraints:**

1. **Spacing:**$\|\mathbf{c}_i - \mathbf{c}_j\| \geq 2r_{\min} = 20 \text{mm}$
2. **Edge margin:**$\text{dist}(\mathbf{c}_i, \partial\Omega) \geq 10 \text{mm}$
3. **Bounds:**$x_{\min} \leq x_i \leq x_{\max}$

**Algorithm:**
```
Initialize: X = [initial grid positions]
for iteration in 1:200:
    # FEA solve
    K = assemble_stiffness(X, geometry)
    u = solve(K, f)
    C = 0.5 * dot(u, K*u)
    
    # Sensitivity analysis
    ∇C = compute_sensitivities(X, K, u)
    
    # MMA update
    X_new = MMA_update(X, C, ∇C, constraints)
    
    # Check convergence
    if |C_new - C| / C < 1e-4:
        break
```

**Expected convergence time:** 20-40 minutes per run

#### 4.3.5. Implementation Resources

**Open-source references:**

- MMC188_python (GitHub: [link](https://github.com/ThomasRochefortB/MMC188_python))
- TopOpt.jl MMC module
- FEniCS topology optimization tutorials

---

## 5. Work Breakdown Structure (WBS)

### Phase 1: Foundation & Dual Environment Setup (Months 1-2)

#### WBS 1.1: Literature Review (Weeks 1-3)
- **Week 1:** Survey differentiable physics (Wu 2024, Lee 2024, Chandrasekhar 2023)
- **Week 2:** Survey PINNs for mechanics (He 2024, Santos 2024, Raissi 2019)
- **Week 3:** Survey MMC topology optimization (Guo 2016, Zhang 2017)
- **Deliverable:** Annotated bibliography (~40 papers, 2020-2025)

#### WBS 1.2: Software Environment Setup (Weeks 4-5)
- **Week 4:** Install Julia + Zygote.jl, PyTorch 2.0+, CUDA toolkit
- **Week 5:** Install Python + FEniCS, Trimesh, Open3D
- **Deliverable:** Verified installations with toy examples

#### WBS 1.3: CAD Processing Pipeline (Week 6)
- Convert STEP files to voxel grids (Trimesh → NumPy)
- Extract surface meshes for screw projection
- Implement load/BC channel encoding
- **Deliverable:** Preprocessed L-bracket benchmark assembly

#### WBS 1.4: Baseline GA Implementation (Week 7-8)
- Implement simple Genetic Algorithm + full FEA
- Run on Benchmark 1 to establish baseline convergence time
- **Deliverable:** GA baseline results (for comparison)

---

### Phase 2: Parallel Implementation (Months 3-4)

#### WBS 2.1: Approach A — Differentiable FEA (Weeks 9-12)
- **Week 9-10:** Implement 2D differentiable FEA (Julia + Zygote.jl)
- **Week 11:** Extend to 3D tetrahedral elements
- **Week 12:** Validate gradients vs. finite differences
- **Deliverable:** Julia module `DiffFEA.jl` with test suite

#### WBS 2.2: Approach A — Gradient-Based Optimization (Weeks 13-14)
- **Week 13:** Implement ADAM optimizer with constraints
- **Week 14:** Test on 2D L-bracket, log training samples
- **Deliverable:** 300-500 sample dataset (HDF5 format)

#### WBS 2.3: Approach B — MMC Implementation (Weeks 9-12)
- **Week 9-10:** Port MMC code, implement 2D compliance minimization
- **Week 11:** Add geometric constraints (spacing, edge margin)
- **Week 12:** Validate on simple cantilever beam
- **Deliverable:** Python module `MMCFasteners.py` with test suite

#### WBS 2.4: Approach B — 3D Extension (Weeks 13-14)
- **Week 13:** Extend MMC to 3D tetrahedral mesh
- **Week 14:** Test on 3D L-bracket benchmark
- **Deliverable:** MMC results on Benchmark 1

---

### Phase 3: PINN Training & Advanced MMC (Months 5-6)

#### WBS 3.1: Approach A — PINN Architecture (Weeks 17-18)
- **Week 17:** Implement 3D CNN encoder + MLP in PyTorch
- **Week 18:** Add physics loss (equilibrium residual)
- **Deliverable:** PINN architecture code

#### WBS 3.2: Approach A — PINN Training (Weeks 19-20)
- **Week 19:** Train on Approach A dataset (300 samples)
- **Week 20:** Hyperparameter tuning (λ values, learning rate)
- **Deliverable:** Trained PINN model (<10% MAE)

#### WBS 3.3: Approach A — PINN-Accelerated Optimization (Week 21)
- Replace FEA with PINN in optimization loop
- Measure speedup vs. Phase 1
- **Deliverable:** Speed comparison plots

#### WBS 3.4: Approach B — Additional Benchmarks (Weeks 19-21)
- Run MMC on Benchmark 2 (motor housing cover)
- Test with different numbers of screws (4, 6, 8)
- **Deliverable:** MMC results on Benchmark 2

#### WBS 3.5: Both Approaches — Generalization Test (Week 22-23)
- **Approach A:** Train on Load Case 1, test on Load Case 2
- **Approach B:** Re-optimize for Load Case 2
- **Deliverable:** Generalization error analysis

---

### Phase 4: Unified Validation & Comparative Analysis (Month 7)

#### WBS 4.1: Ansys Validation (Week 25-26)
- Export optimized screw positions from both methods to Ansys
- Run high-fidelity FEA (nonlinear contact, friction)
- Compare predicted compliance vs. Ansys ground truth
- **Deliverable:** Validation report (error <15% for both methods)

#### WBS 4.2: Comparative Metrics (Week 27)
- Generate unified comparison tables:
  - Convergence time (wall-clock)
  - Final compliance values
  - Screw coordinates (spatial distribution)
  - Implementation complexity (LOC, dependencies)
- **Deliverable:** Comprehensive comparison table

#### WBS 4.3: Visualization & Analysis (Week 28)
- 3D renderings of optimized screw placements (ParaView/Blender)
- Convergence plots overlaid (PINN vs. MMC vs. GA)
- Trade-off diagrams (speed vs. accuracy, training cost vs. inference)
- **Deliverable:** Publication-ready figures

---

### Phase 5: Thesis Writing & Dissemination (Month 8)

#### WBS 5.1: Thesis Document (Weeks 29-31)
- **Week 29:** Draft Chapters 1-3 (Intro, Lit Review, Theoretical Framework)
- **Week 30:** Draft Chapters 4-6 (Methodology, Implementation, Results)
- **Week 31:** Draft Chapters 7-8 (Discussion, Conclusion)
- **Deliverable:** Complete thesis draft (~100 pages)

#### WBS 5.2: Supervisor Revisions (Week 32)
- Incorporate feedback from Dr. Rusli
- Proofread and format
- **Deliverable:** Final thesis PDF

#### WBS 5.3: Conference Paper Preparation (Week 32)
- Extract 8-10 page paper for ASME IDETC or similar
- Target: *Computer-Aided Design*, *Structural & Multidisciplinary Optimization*
- **Deliverable:** Submission-ready manuscript

#### WBS 5.4: Oral Defense Preparation (Week 32)
- Prepare 25-slide presentation
- Video demonstration of both frameworks
- **Deliverable:** Defense slides + demo video

---

## 6. Gantt Chart

| Phase | Month 1 | Month 2 | Month 3 | Month 4 | Month 5 | Month 6 | Month 7 | Month 8 |
|-------|---------|---------|---------|---------|---------|---------|---------|---------|
| **1. Foundation** | ████████ | ████████ |  |  |  |  |  |  |
| **2A. PINN Implementation** |  |  | ████████ | ████████ |  |  |  |  |
| **2B. MMC Implementation** |  |  | ████████ | ████████ |  |  |  |  |
| **3A. PINN Training** |  |  |  |  | ████████ | ████ |  |  |
| **3B. MMC Benchmarks** |  |  |  |  | ████████ | ████ |  |  |
| **4. Validation & Analysis** |  |  |  |  |  |  | ████████ |  |
| **5. Writing/Defense** |  |  |  |  |  |  | ████ | ████████ |

**Legend:** Full month = ████████, Half month = ████

**Critical Path:**

- Months 1-2: Foundational work (literature, setup)
- Months 3-4: Parallel implementation (can work simultaneously on both)
- Months 5-6: Advanced features (PINN training || MMC extended benchmarks)
- Month 7: Unified validation and comparative analysis
- Month 8: Writing and defense preparation

---

## 7. Expected Outcomes & Success Metrics

### 7.1. Technical Deliverables

**Open-Source Software:**

1. **DiffFEA.jl** — Julia differentiable FEA library (~300 LOC)
2. **ScrewOptimPINN.jl** — ADAM optimizer + data logging (~200 LOC)
3. **PINNFastener.py** — PyTorch PINN implementation (~500 LOC)
4. **MMCFasteners.py** — MMC topology optimization (~400 LOC)
5. **ValidationPipeline.py** — Unified Ansys export + comparison (~200 LOC)

**Trained Models:**

- PINN checkpoint (.pth file) for L-bracket benchmark
- Training logs and hyperparameter configs

**Benchmark Results:**

- Raw data files (HDF5) for all experiments
- Preprocessed CAD assemblies (STEP + voxel grids)

### 7.2. Performance Targets

#### Comparison Table

| Metric | GA Baseline | Approach A (PINN) | Approach B (MMC) |
|--------|-------------|-------------------|------------------|
| **Convergence Time** | 2-4 hours | Phase 1: 10-30 min<br>Phase 2: 2-5 min (after training) | 20-40 minutes |
| **Training Overhead** | N/A | 2-4 hours (one-time) | N/A |
| **Iterations to 5% Optimum** | 5000-10,000 | 100-300 | 150-400 |
| **Final Compliance (L-bracket)** | Baseline | ±5% | ±5% |
| **Generalization (Load Case 2)** | Re-run GA | ±15% error | Re-optimize (20-40 min) |
| **Implementation LOC** | ~500 Python | ~1000 (Julia + Python) | ~600 Python |
| **Languages/Dependencies** | Python, FEniCS | Julia, Python, PyTorch | Python, FEniCS |
| **Debugging Complexity** | Low | High (AD + neural net) | Medium |

#### Hypothesis Validation

| Hypothesis | Expected Result | Validation Method |
|------------|-----------------|-------------------|
| H1: PINN is 10-100× faster after training | ✓ Likely confirmed | Wall-clock time comparison |
| H2: Similar compliance (within 10%) | ✓ Likely confirmed | Direct numerical comparison |
| H3: MMC more regular layouts | ? To be determined | Visual inspection + regularity metrics |
| H4: PINN generalizes with 10-20% error | ✓ Likely confirmed | Load Case 2 testing |
| H5: MMC simpler to implement | ✓ Likely confirmed | LOC, debug time, dependency count |

### 7.3. Novelty Claims (Revisited)

1. **Primary Contribution:** First direct, empirical comparison of geometric (MMC) vs. learning-based (PINN) for discrete fastener placement
2. **Methodological Contribution:** Unified validation protocol for hybrid optimization methods
3. **Practical Contribution:** Open-source implementations of both paradigms
4. **Scientific Contribution:** Trade-off analysis revealing applicability domains

### 7.4. Publishability Assessment

**Journal Targets (Primary):**

- *Computer-Aided Design* (IF: 3.5) — "Comparative study" angle
- *Structural and Multidisciplinary Optimization* (IF: 4.2) — Strong fit
- *Engineering Applications of Artificial Intelligence* (IF: 8.0) — PINN angle

**Conference Targets (Fallback):**

- ASME IDETC (International Design Engineering Technical Conferences)
- WCCM (World Congress on Computational Mechanics)

**Estimated Acceptance Probability:** 70-80% (comparative study + open-source code + solid validation)

---

## 8. Risk Management & Mitigation

### 8.1. Risk Matrix

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| **PINN fails to converge** | Medium | Medium | Fallback: Approach B (MMC) alone is publishable |
| **MMC implementation issues** | Low | Medium | Fallback: Use existing MMC188_python with minimal mods |
| **Both methods fail** | Very Low | High | Use Phase 1 (Diff FEA only) as single-method thesis |
| **Timeline overrun** | Medium | Medium | Reduce benchmarks from 2 to 1, skip generalization test |
| **Validation shows >20% error** | Low | Medium | Acknowledge in limitations, focus on relative comparison |
| **Julia AD too slow** | Low | Medium | Switch to JAX (Python) with existing differentiable FEM |
| **PINN training stalls** | Medium | Low | Use simpler MLP (no CNN), increase training data to 500 |
| **Ansys license unavailable** | Low | Low | Use open-source FEA (CalculiX, Code_Aster) for validation |

### 8.2. Fallback Thesis Options

#### Fallback 1: MMC-Only Thesis (If PINN Fails)
**Title:** "Moving Morphable Components for Discrete Fastener Optimization"  

**Content:** 

- Remove Approach A entirely
- Expand MMC validation (3 benchmarks, modal frequency objective)
- Still novel: First MMC for fasteners
- **Publishability:** Conference paper (ASME IDETC)

#### Fallback 2: Differentiable Physics Only (If PINN Fails)
**Title:** "Differentiable Physics for Gradient-Based Discrete Fastener Placement"  

**Content:**

- Focus on Phase 1 (Approach A without PINN)
- Compare ADAM vs. L-BFGS vs. CMA-ES optimizers
- Still novel: First gradient-based discrete placement
- **Publishability:** Conference paper (IDETC or SMO)

#### Fallback 3: Single Benchmark Comparison (If Timeline Too Tight)
- Run both methods on L-bracket only
- Skip motor housing cover (Benchmark 2)
- Skip generalization test
- Still provides comparative insight
- **Publishability:** Workshop paper or thesis-only

---

## 9. Benchmark Assemblies

### 9.1. Benchmark 1: L-Bracket (Simplified)

**Geometry:**

- Vertical plate: 100mm × 80mm × 5mm
- Horizontal plate: 80mm × 60mm × 5mm
- Material: Aluminum (E = 70 GPa, ν = 0.3)

**Loading:**

- Load Case 1: 1000N downward at free end of horizontal plate
- Load Case 2: 500N lateral load on vertical plate

**Boundary Conditions:**

- Fixed: Back face of vertical plate

**Optimization:**

- Place N = 4 screws to minimize compliance
- Constraints: Min spacing 20mm, edge margin 10mm

**Why this benchmark:**

- Standard in topology optimization literature
- Simple enough for 2D prototyping
- Extends naturally to 3D

### 9.2. Benchmark 2: Motor Housing Cover (Realistic)

**Geometry:**

- Circular plate: Diameter 150mm, thickness 8mm
- Raised boss: Diameter 40mm, height 15mm
- Material: Cast iron (E = 120 GPa, ν = 0.27)

**Loading:**

- Internal pressure: 0.5 MPa
- Bolt pretension: 2000N per screw

**Boundary Conditions:**

- Fixed: Outer ring (Ø150mm)

**Optimization:**

- Place N = 6-8 screws around boss perimeter
- Constraints: Angular spacing ≥ 30°, radial position 50-70mm

**Why this benchmark:**

- Realistic industrial application
- Tests performance on non-trivial geometry
- Validation possible with experimental data from literature

---

## 10. Key References (40 Core Papers)

### Differentiable Physics (12 papers)
1. Wu, G. (2024). "JAX-SSO: Differentiable FEA for structural optimization." *arXiv:2407.20026*
2. Lee, K.J. et al. (2024). "Differentiable structural analysis framework." *Automation in Construction*
3. Chen, Z. & Shen, Y. (2020). "Topology optimization through differentiable FEM." *arXiv:2009.10072*
4. Chandrasekhar, A. (2023). "Differentiable physics for topology optimization." *Comput. Methods Appl. Mech. Eng.*
5. Ambrozkiewicz, O. & Kriegesmann, B. (2020). "Simultaneous topology & fastener layout." *arXiv:2005.03398*
6. Hoyer, S. et al. (2024). "Neural networks + differentiable FEA." *arXiv preprint*
7. Bollapragada, R. (2024). "Adjoint sensitivities for structural design." *SIAM J. Sci. Comput.*
8. Giles, M. (2008). "Adjoint methods in CFD." *J. Comput. Phys.*
9. Griewank, A. (2008). *Evaluating Derivatives: Principles of Automatic Differentiation* (Textbook)
10. Pastrana, R. et al. (2025). "Physics-constrained neural networks for optimization." *ICLR 2025*
11. Tortorelli, D.A. & Michaleris, P. (1994). "Design sensitivity analysis." *Comput. Methods Appl. Mech. Eng.*
12. Bendsøe, M.P. & Sigmund, O. (2003). *Topology Optimization: Theory, Methods, and Applications* (Textbook)

### Physics-Informed Neural Networks (12 papers)
13. Raissi, M. et al. (2019). "Physics-informed neural networks." *J. Comput. Phys. 378:686-707*
14. He, Q. et al. (2024). "PINN for structural mechanics." *Nature Communications*
15. Santos, L. (2024). "Deep and PINNs for FEA acceleration." *ICMLT 2024*
16. Karniadakis, G.E. et al. (2021). "Physics-informed machine learning." *Nature Reviews Physics*
17. Bastek, J.H. et al. (2023). "Energy-based PINNs for elasticity." *Comput. Methods Appl. Mech. Eng.*
18. Cai, S. et al. (2021). "Physics-informed neural networks for beam dynamics." *ASME J. Appl. Mech.*
19. Lu, L. et al. (2021). "DeepXDE: Deep learning library for PDEs." *SIAM Review*
20. Grossmann, T. et al. (2023). "Can PINNs beat FEM?" *arXiv:2302.04107*
21. Wang, C. et al. (2024). "Data-assisted PINNs for structures." *Int. J. Mech. Sci.*
22. Gong, Y. et al. (2025). "PIKAN for multi-material elasticity." *arXiv preprint*
23. Leng, K. & Thiyagalingam, J. (2022). "On compatibility between NNs and PDEs." *arXiv:2212.00270*
24. Jeong, H. et al. (2023). "Complete PINN-based topology optimization." *Comput. Methods Appl. Mech. Eng.*

### Moving Morphable Components (8 papers)
25. Guo, X. et al. (2014). "Doing topology optimization explicitly and geometrically—a new moving morphable components based framework." *J. Appl. Mech. 81(8)*
26. Zhang, W. et al. (2017). "A new topology optimization approach based on MMC." *Struct. Multidiscip. Optim. 53(6):1243-1260*
27. Zhang, W. et al. (2016). "Explicit layout control in optimal design of structural systems with multiple embedding components." *Comput. Methods Appl. Mech. Eng.*
28. Norato, J.A. et al. (2015). "A geometry projection method for continuum-based topology optimization." *Int. J. Numer. Methods Eng.*
29. Guo, X. et al. (2016). "Explicit structural topology optimization based on MMC." *Comput. Methods Appl. Mech. Eng.*
30. Zhou, Y. et al. (2020). "Feature-driven topology optimization method with signed distance function." *Comput. Methods Appl. Mech. Eng.*
31. Hoang, V.N. & Jang, G.W. (2017). "Topology optimization using moving morphable bars." *Comput. Methods Appl. Mech. Eng.*
32. Lei, X. et al. (2019). "Machine learning-driven real-time topology optimization." *arXiv preprint*

### Surrogate Modeling & Comparative Studies (8 papers)
33. Fu, J. et al. (2023). "GNN surrogate for FEA." *J. Comput. Design Eng.*
34. Gladstone, R. et al. (2024). "Deep surrogates for PDEs." *Nature Sci. Rep.*
35. Moran, J.A. & Morato, P.G. (2025). "Active learning for reliability." *arXiv:2508.18170*
36. Hinrichsen, J. et al. (2024). "Dropout active learning for FEA." *Front. Physiol.*
37. de Avila Belbute-Peres, F. et al. (2024). "HyperPINN: Parameterized differential equations with hypernetworks." *ICML 2024*
38. Groen, J.P. (2018). "Multi-scale design methods for topology optimization." *PhD Thesis, DTU*
39. EL Ghadoui, M. et al. (2023). "Hybrid optimization approach for intelligent manufacturing." *Nature Sci. Rep.*
40. Rusli, L.P. (2008). "Kinematic screw theory for assemblies." *PhD Dissertation, Ohio State*

---

## 11. Thesis Structure (Projected ~100-110 Pages)

### Chapter 1: Introduction (10 pages)
1.1. Motivation: The Discrete Fastener Placement Problem  
1.2. Current Approaches and Limitations  
1.3. Research Gap: Lack of Comparative Studies  
1.4. Thesis Contributions & Novelty  
1.5. Thesis Roadmap & Reading Guide

### Chapter 2: Literature Review (18 pages)
2.1. Topology Optimization: From Density Methods to Geometric Frameworks  
2.2. Differentiable Physics in Structural Design (2020-2025)  
2.3. Physics-Informed Neural Networks for Mechanics  
2.4. Moving Morphable Components: Theory and Applications  
2.5. Surrogate Modeling and Active Learning  
2.6. Discrete Fastener Optimization (Medical and Mechanical)  
2.7. Research Positioning & Gap Analysis  
2.8. Comparative Study Motivation

### Chapter 3: Theoretical Framework (16 pages)
3.1. Linear Elasticity and Finite Element Method  
3.2. Compliance Minimization as Structural Objective  
3.3. Automatic Differentiation: Forward and Reverse Modes  
3.4. Adjoint Sensitivity Analysis for FEA  
3.5. Physics-Informed Neural Networks: Mathematical Foundation  
3.6. Moving Morphable Components: Geometric Parameterization  
3.7. Active Learning Paradigm

### Chapter 4: Methodology (24 pages)
4.1. System Architecture Overview  
4.2. Common Infrastructure (CAD Processing, Validation Protocol)  
4.3. **Approach A: Hybrid Differentiable Physics + PINN**  
   4.3.1. Differentiable FEA Implementation (Julia + Zygote.jl)  
   4.3.2. Gradient-Based Screw Placement (Phase 1)  
   4.3.3. PINN Surrogate Architecture & Training (Phase 2)  
   4.3.4. PINN-Accelerated Optimization (Phase 3)  
4.4. **Approach B: Moving Morphable Components**  
   4.4.1. Geometric Component Parameterization  
   4.4.2. Material Projection and Ersatz Model  
   4.4.3. FEA and Sensitivity Analysis (FEniCS)  
   4.4.4. MMA Optimization Algorithm  
4.5. Benchmark Assemblies and Test Cases  
4.6. Unified Validation Protocol (Ansys Export)

### Chapter 5: Implementation (12 pages)
5.1. Software Stack and Dependencies  
5.2. **Approach A Implementation Details**  
   5.2.1. DiffFEA.jl Module Structure  
   5.2.2. PINN Training Pipeline (PyTorch)  
   5.2.3. Data Logging and Preprocessing  
5.3. **Approach B Implementation Details**  
   5.3.1. MMCFasteners.py Module Structure  
   5.3.2. Constraint Handling and Projection  
5.4. Computational Resources and Hardware  
5.5. Code Availability and Reproducibility

### Chapter 6: Results (20 pages)
6.1. Benchmark 1: L-Bracket Assembly  
   6.1.1. Approach A Results (Phase 1, 2, 3)  
   6.1.2. Approach B Results  
   6.1.3. GA Baseline Results  
   6.1.4. Convergence Comparison  
   6.1.5. Final Screw Placements and Compliance  
6.2. Benchmark 2: Motor Housing Cover (if time permits)  
   6.2.1. Approach A Results  
   6.2.2. Approach B Results  
   6.2.3. Comparative Analysis  
6.3. Generalization Tests  
   6.3.1. Approach A: Load Case 2 (PINN without retraining)  
   6.3.2. Approach B: Load Case 2 (Re-optimization)  
6.4. Ansys Validation Results  
   6.4.1. High-Fidelity FEA Comparison  
   6.4.2. Error Analysis

### Chapter 7: Comparative Discussion (14 pages)

7.1. Performance Comparison Summary  
7.2. **Convergence Speed Trade-Offs**  
   7.2.1. Training Cost vs. Inference Speed (PINN)  
   7.2.2. Direct Optimization Cost (MMC)  
   7.2.3. Breakeven Analysis (When is PINN Overhead Justified?)  
7.3. **Solution Quality and Layout Characteristics**  
   7.3.1. Compliance Values  
   7.3.2. Spatial Distribution Regularity  
   7.3.3. Manufacturability Considerations  
7.4. **Generalization Capability**  
   7.4.1. PINN Transfer Learning  
   7.4.2. MMC Re-Optimization Cost  
7.5. **Implementation Complexity**  
   7.5.1. Lines of Code, Dependencies  
   7.5.2. Debugging Challenges  
   7.5.3. Learning Curve for Practitioners  
7.6. **Applicability Domains**  
   7.6.1. When to Use PINN (Design Families, Repeated Queries)  
   7.6.2. When to Use MMC (One-Off Designs, Geometric Control)  
7.7. Limitations of Both Approaches  
7.8. Answers to Research Questions (RQ1-RQ5)

### Chapter 8: Conclusion (6 pages)
8.1. Summary of Contributions  
8.2. Key Findings and Recommendations  
8.3. Implications for Engineering Practice  
8.4. Future Work  
   8.4.1. Hybrid PINN-MMC Framework  
   8.4.2. Extension to Contact Mechanics  
   8.4.3. Multi-Material Assemblies  
8.5. Closing Remarks

### Appendices (12 pages)
A. Derivation of Adjoint Sensitivities  
B. PINN Loss Function Details and Hyperparameters  
C. MMC Mathematical Formulation  
D. Complete Code Repository Structure  
E. Benchmark Assembly CAD Files (STEP)  
F. Experimental Data Tables  
G. Additional Convergence Plots

### References (~100-120 entries)

---

## 12. Budget Estimate

| Item | Specification | Unit Cost (IDR) | Quantity | Total (IDR) |
|------|---------------|-----------------|----------|-------------|
| **Cloud GPU (PINN Training)** | NVIDIA RTX 3060 equiv. (Lambda Labs) | 300,000/hour | 10 hours | 3,000,000 |
| **Cloud CPU (MMC Runs)** | 8-core CPU instance | 100,000/hour | 5 hours | 500,000 |
| **Ansys License** | Student or dept. license | 0 | - | 0 |
| **CAD Software** | Inventor or SolidWorks (student/dept. license) | 0 | - | 0 |
| **Conference Fee** | ASME IDETC student registration | 3,000,000 | 1 | 3,000,000 |
| **Journal OA Fee** | *Struct. Opt.* student discount (optional) | 8,000,000 | 0.5 | 4,000,000 |
| **Miscellaneous** | Documentation, printing | - | - | 500,000 |
| **Total (Conference Only)** | | | | **7,000,000** |
| **Total (With Journal)** | | | | **11,000,000** |

**Note:** GPU training can be done on personal hardware if available (NVIDIA RTX 2060+ sufficient).

---

## 13. Timeline Compression Strategies

### 13.1. If Timeline is Tight (7 Months Instead of 8)

**Reduction Strategy:**

- Skip Benchmark 2 (motor housing cover) → Focus only on L-bracket
- Skip generalization test (Load Case 2)
- Reduce PINN training samples to 200-300
- Skip multi-optimizer comparison (use ADAM only for both)

**Result:** Still produces publishable comparative study on single benchmark

### 13.2. If One Approach Fails Mid-Thesis

**Fallback Path:**

- If PINN fails by Month 5 → Expand MMC to 3 benchmarks, add modal frequency objective
- If MMC fails by Month 5 → Expand PINN to include GNN surrogate comparison
- Either fallback still yields novel single-method thesis

---

## 14. Success Criteria for Bachelor Thesis

### 14.1. Minimum Requirements for Graduation

✓ At least **one method** (PINN or MMC) successfully optimizes discrete fasteners on 1 benchmark  
✓ Validation against commercial FEA (Ansys) with error <20%  
✓ Comparison against GA baseline showing improvement  
✓ Complete thesis document (~80+ pages)  
✓ Oral defense with working demonstration

### 14.2. Target Requirements for Strong Publication

✓ **Both methods** successfully implemented on 2 benchmarks  
✓ Direct comparison of convergence speed, solution quality  
✓ Generalization test on unseen load case  
✓ Open-source code release  
✓ Validation error <15%  
✓ Comprehensive discussion of trade-offs

### 14.3. Stretch Goals (If Time Permits)

✓ Third benchmark assembly (industry-provided CAD)  
✓ Hybrid PINN-MMC framework (use MMC as initialization for PINN)  
✓ Experimental validation on 3D-printed assemblies  
✓ Conference presentation accepted

---

## 15. Collaboration & Supervision Strategy

### 15.1. Supervision Structure & Involvement

#### Primary Supervisor: Dr. Leonard P. Rusli

**Role:** Principal research advisor, methodology oversight, thesis evaluation  
**Position:** Vice Rector for Academic Affairs, Swiss German University  
**Expected Availability:** Moderate (administrative duties at SGU, estimated 3-5 hours/week)

**Meeting Schedule:**

- Bi-weekly progress meetings (30 min, in-person at SGU or virtual)
- Monthly deep-dive reviews (1 hour)
- Critical milestones: Phase transitions, validation results, thesis drafts

**Primary Supervisor Responsibilities:**

- High-level research direction and novelty assessment
- Critical feedback on the dual-method comparative framework
- Thesis draft review (2 rounds: draft + final)
- Defense preparation and examination
- Final thesis approval

#### Co-Supervisor (Prospective): Dr. Eka Budiarto, S.T., M.Sc.

**Role:** Technical consultant for mathematical methods and computational implementation  
**Position:** Head of Master of Information Technology, Swiss German University  
**Background:** Mathematics and Information Technology (IT)  
**Expected Availability:** Currently uncertain; casual consultation phase (Nov 2025 – Dec 2025)

**Status Timeline:**

- **Current (Nov–Dec 2025):** Informal technical consultation  
  - Methodology comparison guidance (already provided critical support in developing the PINN vs. MMC architectural approaches)  
  - Mathematical formulation review  
  - Literature review assistance (comparative analysis frameworks)  
- **Potential (Jan 2026 onwards):** Official co-supervisor designation  
  - Subject to availability confirmation when the new semester begins  
  - Would transition to formal co-supervision role if schedule permits  

**Co-Supervisor Focus Areas (if confirmed):**

- Mathematical rigor of PINN loss functions and MMC formulations
- Computational implementation guidance (Julia AD, PyTorch, FEniCS)
- Comparative methodology design and statistical analysis
- Algorithm debugging and optimization convergence issues
- Code review and software engineering best practices

**Meeting Schedule (if co-supervisor role confirmed):**

- Bi-weekly technical deep-dives (45–60 min, in-person at SGU)
- Ad-hoc consultations for debugging critical issues
- Code review sessions at key milestones (Months 4, 6, 7)

#### Supervision Strategy

**Decision-Making Protocol:**

- **Research direction & scope:** Dr. Rusli (primary supervisor)
- **Technical implementation details:** Dr. Budiarto (if co-supervisor)
- **Major methodology changes:** Joint consultation with both supervisors
- **Timeline adjustments:** Approval by Dr. Rusli

**Communication Channels:**

- Email for non-urgent updates (weekly progress summaries)
- WhatsApp/Telegram for quick technical questions (response target: 24–48 hours)
- Zoom/Teams for virtual meetings
- In-person meetings at SGU campus (primary venue, since both supervisors are SGU faculty)

#### Collaborative Advantages of Dual Supervision

**Complementary Expertise:**

- **Dr. Rusli:** Mechanical engineering, screw theory, research strategy, academic leadership
- **Dr. Budiarto:** Mathematical methods, computational optimization, machine learning, software engineering

**Existing Collaboration:**

- Dr. Budiarto has already contributed significantly to the methodology comparison that led to the two architectural approaches.
- His IT and mathematics background has been helpful for translating mechanical engineering ideas into concrete computational architectures.

**Risk Mitigation:**

- If Dr. Rusli has limited availability due to administrative duties → Dr. Budiarto can provide technical continuity.
- If Dr. Budiarto cannot commit as co-supervisor → Primary supervision by Dr. Rusli plus the self-sufficiency plan remains sufficient.
- Dual perspectives strengthen methodology validation and novelty claims.

**Enhanced Thesis Quality:**

- Mathematical rigor (Dr. Budiarto) plus engineering relevance (Dr. Rusli) supports a strong comparative study.
- IT background supports dual-language implementation (Julia + Python).
- Mathematics background supports correct formulation of PINN physics losses and MMC sensitivities.
- Both supervisors being SGU faculty enables frequent in-person consultation.

### 15.2. Self-Sufficiency Plan

**Resources:**

- Extensive open-source code references (MMC188_python, JAX-SSO, DeepXDE)
- Online communities: Julia Discourse, PyTorch Forums, FEniCS Q&A
- Literature: 40 core papers provide detailed implementation guidance

**Learning Strategy:**

- Week 1-2: Intensive Julia/Zygote.jl tutorial (differentiable programming)
- Week 3-4: PyTorch PINN tutorials (Stanford CS230, DeepXDE docs)
- Week 5-6: FEniCS topology optimization examples

**Documentation:**

- Maintain detailed engineering notebook (Markdown + Jupyter)
- Weekly self-reflection on progress vs. timeline
- Early warning system: Flag issues 2 weeks before critical path impact

---

## 16. Conclusion: Why This Dual Approach Will Succeed

### 16.1. Technical Soundness

✓ **Proven components:** Both PINN and MMC have successful implementations in literature  
✓ **Independent fallback:** Either method alone suffices for graduation  
✓ **Shared infrastructure:** Benchmarks, validation, preprocessing can be reused  
✓ **Tractable scope:** Reduced from original 10k-sample surrogate to 300-500 samples

### 16.2. Timeline Realism

✓ **Parallel work:** PINN and MMC implementations do not block each other (Months 3-4)  
✓ **Critical path overlap:** Only validation phase requires both methods complete (Month 7)  
✓ **Built-in buffer:** Writing in Month 8 has 1-month flexibility if needed  
✓ **Fallback options:** Clear downscoping strategies if timeline tightens

### 16.3. Scientific Impact

✓ **Unique contribution:** First direct comparison of geometric vs. learning-based for discrete fasteners  
✓ **Practical value:** Trade-off analysis guides engineering practitioners  
✓ **Open science:** Reproducible implementations benefit research community  
✓ **Publishability:** Journal-grade comparative study (not just method demonstration)

### 16.4. Risk Mitigation

✓ **Dual fallback:** Either PINN or MMC alone is publishable  
✓ **Benchmark reduction:** Can drop from 2 to 1 benchmark if needed  
✓ **Validation flexibility:** Open-source FEA if Ansys unavailable  
✓ **Supervisor bandwidth:** Strong self-sufficiency plan with open-source resources

### 16.5. Personal Alignment (Your Profile)

✓ **Mechatronics background:** Comfortable with both ML and mechanics  
✓ **Computational focus:** Strong coding skills (Julia + Python feasible)  
✓ **Timeline:** 8 months with June 2026 deadline accommodates dual approach  
✓ **Ambition:** Comparative study maximizes thesis impact for future opportunities

---

## 17. Final Remarks

This dual-approach thesis plan transforms a **method demonstration** into a **methodology comparison study**—a significantly stronger scientific contribution. The key innovation is not in proving one method superior, but in **revealing the trade-offs** that guide practitioners toward the right tool for their specific problem.

**The Path Forward:**

✅ **Months 1-2:** Foundational work (literature, dual environment setup, benchmarks)  
✅ **Months 3-4:** Parallel implementation (PINN + MMC, independent tracks)  
✅ **Months 5-6:** Advanced features (PINN training, MMC extended tests)  
✅ **Month 7:** Unified validation and comparative analysis  
✅ **Month 8:** Writing and defense (with 2-3 week buffer)

**Success Metrics:**

- Minimum: **One method** on **one benchmark** → Bachelor thesis (graduated)
- Target: **Both methods** on **two benchmarks** → Strong publication (journal paper)
- Stretch: **Both methods** + **generalization** + **open-source release** → Exceptional thesis (conference talk + journal)

**This thesis has the potential to define the state-of-the-art in discrete fastener optimization while maintaining graduation safety through built-in fallback options. The dual approach is ambitious yet feasible—exactly what a strong bachelor thesis should be.**

---

**Document Version:** 1.0  
**Date:** November 21, 2025  
**Prepared by:** AI Research Advisor (Perplexity)  
**Status:** Ready for supervisor review and approval  
**Next Steps:** Schedule meeting with Dr. Rusli to finalize approach selection

---

## Appendix: Quick Decision Matrix

### Should You Pursue the Dual Approach?

**YES, pursue dual approach if:**

- ✓ You want journal-level publication (not just conference)
- ✓ You can commit 45% extra effort (realistic for full-time thesis)
- ✓ You are comfortable debugging both Julia and Python
- ✓ Your goal is maximizing research impact
- ✓ Supervisor values comparative methodology over single-method depth

**NO, stick to single approach if:**

- ✓ Timeline is absolutely non-negotiable
- ✓ You strongly prefer one tech stack over the other
- ✓ Your primary goal is graduation, not publication
- ✓ Computational resources are very limited
- ✓ You prefer deep exploration of one method over breadth

**Recommended:** **DUAL APPROACH** (with scope reductions outlined in Section 13)

The scientific value gain justifies the 45% effort increase, and the mutual fallback safety ensures graduation even if one method fails. This is a calculated risk with high reward potential.
