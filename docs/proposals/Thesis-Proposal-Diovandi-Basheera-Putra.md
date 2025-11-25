# COMPARATIVE EVALUATION OF GEOMETRIC AND LEARNING-BASED OPTIMIZATION FRAMEWORKS FOR DISCRETE FASTENER PLACEMENT

**By**  
**Diovandi Basheera Putra**  
**Student ID Number: 12301058**

**BACHELOR'S DEGREE**  
**in**  
**MECHANICAL ENGINEERING - MECHATRONICS**

**FACULTY OF ENGINEERING AND INFORMATION TECHNOLOGY**

**SWISS GERMAN UNIVERSITY**  
The Prominence Tower  
Jalan Jalur Sutera Barat No. 15, Alam Sutera  
Tangerang, Banten 15143 - Indonesia

**November 2025**


---

## PROPOSED ADVISORS

I propose **Dr. Leonard P. Rusli, B.Sc, M.Sc, Ph.D** as my Advisor for my proposed Thesis.

---

## CHAPTER 1 - INTRODUCTION

### 1.1 Background

Mechanical assemblies in automotive, aerospace, and manufacturing industries rely fundamentally on discrete fasteners such as screws, bolts, and rivets to ensure structural integrity[1][41][44]. These fastening elements serve three critical functions: removing degrees of freedom between rigid bodies, transferring loads across interfaces, and enabling assembly and disassembly operations[1]. Despite their ubiquity, the placement of these fasteners remains largely a manual process driven by engineering heuristics, trial-and-error prototyping, and post-hoc validation through Finite Element Analysis (FEA)[1][32][44].

Current industrial practice typically follows empirical rules such as "place screws every 50mm" or positioning fasteners symmetrically around load-bearing regions[1][39]. While these approaches have proven adequate for conventional designs, they fail to leverage computational optimization methods that could systematically minimize structural compliance (maximize stiffness) while reducing material usage and assembly complexity[1][41]. The inefficiency of manual fastener placement becomes particularly pronounced in modern design contexts where lightweighting, cost reduction, and rapid prototyping are paramount[33][39].

Recent advances in computational mechanics from 2020 to 2025 have introduced two fundamentally different paradigms for addressing discrete placement optimization problems[4][5][13]. The first paradigm, **learning-based optimization**, leverages differentiable physics simulators combined with Physics-Informed Neural Networks (PINNs) to create ultra-fast surrogate models capable of predicting structural performance in milliseconds[5][13][22][25]. These methods employ automatic differentiation to compute gradients of compliance with respect to continuous fastener coordinates, enabling gradient-based optimization through frameworks such as JAX and Julia/Zygote[4][5][12].

The second paradigm, **geometric optimization**, employs explicit parameterization methods such as Moving Morphable Components (MMC) to represent each fastener as a geometric entity with continuous design variables (position, orientation)[14][17][31][92]. MMC methods optimize these parameters directly through adjoint sensitivity analysis coupled with gradient-based optimizers such as the Method of Moving Asymptotes (MMA)[17][31][92][95]. Unlike density-based topology optimization, MMC maintains crisp geometric boundaries throughout the optimization process, facilitating direct translation to Computer-Aided Design (CAD) models[14][17][92].

Despite these parallel developments, **no existing literature directly compares geometric topology optimization (MMC) with physics-informed machine learning (PINN) for discrete mechanical fastener placement**[1]. This critical gap prevents practitioners from making informed decisions about which computational framework to adopt for their specific design scenarios. The absence of comparative empirical evidence leaves fundamental questions unanswered: Which method converges faster? Do they produce equivalent solutions? Can learning-based methods generalize to unseen load cases? What are the implementation trade-offs in terms of software dependencies, debugging complexity, and computational resources?[1]

The motivation for this comparative study extends beyond academic curiosity. As manufacturing industries increasingly adopt generative design and AI-driven engineering tools[42][45][51], understanding the strengths, weaknesses, and applicability domains of competing optimization paradigms becomes essential for effective technology transfer. This thesis aims to provide the first comprehensive empirical comparison of these two methodologies on standardized benchmark assemblies, thereby establishing a foundation for informed adoption in both research and industrial contexts[1][41].

### 1.2 Problem Statement

**Core Research Problem:**  
How do learning-based (Differentiable Physics + PINN) and geometric (MMC) optimization frameworks compare in terms of convergence speed, solution quality, generalization capability, and implementation feasibility for discrete fastener placement in mechanical assemblies?[1]

This comparative study addresses a significant gap in the structural optimization literature, where PINNs and MMC methods have been developed and validated independently but never systematically compared on identical benchmark problems[13][25][31][92]. Existing PINN studies focus on continuum topology optimization without addressing discrete component placement[13][19][22][25], while MMC literature emphasizes geometric parameterization flexibility without exploring machine learning alternatives[14][17][31][92].

The problem is further complicated by the fact that these methods operate under fundamentally different assumptions and computational paradigms. PINN-based approaches require substantial upfront training investment (2-4 hours) to achieve ultra-fast inference (~1ms per evaluation)[1][5], making them potentially suitable for design families where amortized cost justifies the training overhead. Conversely, MMC methods require no training data but incur direct FEA cost per iteration (~20-40 minutes per optimization run)[1][31], suggesting applicability to one-off design problems. Without empirical evidence, engineers cannot assess these trade-offs quantitatively[1].

### 1.3 Research Questions

This study investigates five specific research questions to guide the comparative analysis:

**RQ1: Convergence Speed**  
Does the PINN surrogate achieve faster optimization convergence than direct MMC optimization? What is the trade-off between one-time PINN training cost (2-4 hours) versus per-iteration MMC cost?[1]

**RQ2: Solution Quality**  
Do both methods converge to similar optimal screw placements and compliance values when applied to identical benchmark assemblies? If solutions differ, which produces more manufacturable layouts that satisfy geometric constraints (minimum spacing, edge margins)?[1]

**RQ3: Generalization Capability**  
When trained on one load case, can the PINN surrogate generalize to unseen load configurations with acceptable error (<15%)? Does MMC require complete re-optimization for each new load case, and what is the computational cost of this re-optimization?[1]

**RQ4: Implementation Feasibility**  
Which method is more tractable for bachelor-level implementation given limited time and computational resources? What are the relative debugging challenges, lines of code (LOC), software dependencies, and hardware requirements?[1]

**RQ5: Applicability Domains**  
For what design scenarios (single design vs. design families, simple vs. complex geometries, 2D vs. 3D) is each method preferred? What practical guidance can be provided to practitioners selecting between these approaches?[1]

### 1.4 Status Update (November 25, 2025)

- **Multi-Geometry PINN:** A 2×96 Tanh MLP trained on 180 differentiable-FEA samples (L-bracket horizontal/vertical, tapered plate combined, ribbed channel upward/shear) achieves ≤1.1 % MAPE on all high-energy loads (3.28 J MAE on L-bracket horizontal, 12.4 J on vertical, 1.55 J on tapered plate, 1.46 J on ribbed channel upward, 2.48 J absolute on ribbed channel shear) while inferring at ~0.009 ms/sample (>1 000× faster than FEA).
- **MMC Baseline:** The refactored `mmc_core.py` + `run_mmc_lbracket.py` pipeline converges in ~30 iterations for L-bracket horizontal/vertical loads with explicit spacing/edge constraints; extensions to the tapered plate and ribbed channel are queued next.
- **Data & Artifacts:** Scenario-validation rollouts (`src/experiments/scenario_validation/rollouts/`) generate the multi-geometry datasets stored in `data/results/multi_geom_training/`, with manifests and MAE tables maintained under `data/results/`.
- **Next Work:** Scale each load case to ≥200 samples per `dataset_plan.yaml`, run MMC on the additional geometries, and execute Ansys/CalculiX validation with the refreshed layouts so Chapters 5–7 include external verification.

### 1.4 Research Objectives

The primary goal of this thesis is to develop and validate both optimization frameworks on identical benchmark mechanical assemblies, producing a comprehensive comparative analysis that guides future research and industrial application of discrete placement optimization[1]. The study pursues five specific objectives:

**O1: Implement Approach A (Differentiable Physics + PINN)**  

- Build a differentiable static FEA solver in Julia using Zygote.jl for automatic differentiation[4][5][12]
- Implement gradient-based screw placement optimizer using ADAM algorithm[1][5]
- Generate 300-500 training samples from optimization trajectories[1][56]
- Train a PyTorch-based PINN to predict compliance from screw configurations[13][22][25]
- Achieve <10% prediction error versus ground truth FEA on validation set[1]

**O2: Implement Approach B (MMC Framework)**  

- Port and adapt MMC topology optimization code using Python and FEniCS[17][31][92]
- Parameterize fasteners as circular/cylindrical geometric components with explicit coordinates[14][31][92]
- Implement geometric constraints (minimum spacing 20mm, edge margin 10mm)[1][31]
- Optimize using Method of Moving Asymptotes (MMA) or SLSQP algorithms[31][92]
- Converge to discrete screw placements in <30 minutes per optimization run[1]

**O3: Establish Unified Validation Protocol**  

- Select two benchmark assemblies: L-bracket and motor housing cover[1]
- Define common performance metrics: compliance, convergence time, screw coordinates[1]
- Export optimized layouts to Ansys for high-fidelity validation[1]
- Compare against baseline Genetic Algorithm + Full FEA to establish reference performance[1]

**O4: Conduct Comprehensive Comparative Analysis**  

- Generate convergence plots comparing compliance versus iteration and wall-clock time[1]
- Compare final screw placements using spatial distribution metrics[1]
- Analyze generalization performance on unseen Load Case 2[1]
- Document implementation complexity: lines of code, debugging time, software stack[1]

**O5: Deliver Open-Source Implementations**  

- Publish Julia DiffFEA module with PINN training scripts[1][4][5]
- Publish Python MMC fastener optimizer[1][17][31]
- Provide benchmark CAD files (STEP format) and validation protocols[1]
- Write comprehensive documentation and tutorials for reproducibility[1]

### 1.5 Scope and Limitations

**Scope:**

This thesis focuses exclusively on **discrete fastener placement optimization for maximizing structural stiffness** (equivalently, minimizing compliance) under static linear elasticity assumptions[1]. The study implements both PINN-based and MMC-based optimization frameworks on two benchmark assemblies: a simplified L-bracket (2D/3D) and a motor housing cover (3D)[1]. The primary load case involves static point loads or distributed forces, with secondary analysis examining generalization to a second load configuration[1].

Fasteners are modeled as **cylindrical regions with enhanced stiffness** (E_bolt = 200 GPa for steel bolts) embedded in base materials (aluminum or cast iron)[1]. Optimization variables include continuous x, y, z coordinates of screw centers, subject to geometric constraints on minimum spacing and edge distance[1][31]. The PINN surrogate employs a 3D Convolutional Neural Network (CNN) architecture trained on 300-500 samples, while MMC uses finite element analysis via FEniCS with tetrahedral meshes[1][13][31].

**Limitations:**

Several simplifications bound the scope of this bachelor-level thesis:

1. **Physics Model:** The study employs **linear elastic FEA** without contact mechanics, friction, or bolt pretension effects[1]. This simplification enables tractable implementation within 8 months but omits nonlinear phenomena present in real bolted joints[43][46][50]. Validation against Ansys includes high-fidelity contact simulation to assess error introduced by these assumptions[1].

2. **Geometry Complexity:** Benchmarks are limited to **moderate-complexity assemblies** (L-bracket: ~10,000 DOF, motor housing: ~50,000 DOF) to maintain reasonable FEA solve times on available hardware (NVIDIA RTX GPU)[1][5]. Industrial assemblies with millions of DOF exceed computational scope.

3. **Number of Fasteners:** Optimization considers **fixed numbers of fasteners** (n=4, 6, or 8) rather than simultaneous topology-and-quantity optimization[1]. This constraint simplifies the discrete-continuous optimization problem but prevents exploration of the optimal number of screws.

4. **Manufacturing Constraints:** While **minimum spacing (20mm) and edge margins (10mm)** are enforced[1][31], advanced manufacturability considerations such as accessibility for assembly tools, clearance for bolt heads, and torque wrench access are not modeled.

5. **Load Cases:** The study examines **two static load cases per benchmark**[1] to assess generalization. Dynamic loads, fatigue, and multi-load scenarios are excluded to maintain implementation feasibility within the thesis timeline.

6. **Software Scope:** PINN implementation uses **Julia (Zygote.jl) + PyTorch**, while MMC uses **Python (FEniCS)**[1][4][5][31]. Integration with commercial CAD systems (SolidWorks, Inventor) is limited to STEP file import/export.

### 1.6 Significance of the Study

This thesis makes several important contributions to both **academic research** and **engineering practice**:

**Academic Contributions:**

1. **First Direct Comparative Study:** To the best of the author's knowledge, this is the first empirical comparison of geometric topology optimization (MMC) with physics-informed machine learning (PINN) for discrete fastener placement[1]. The study fills a critical gap in the structural optimization literature where these methods have evolved in parallel without systematic benchmarking[13][25][31][92].

2. **Unified Validation Protocol:** By establishing standardized benchmark assemblies, performance metrics, and validation procedures applicable to both paradigms, this work provides a methodological foundation for future comparative studies in computational mechanics[1].

3. **Generalization Analysis:** The thesis contributes empirical evidence on PINN generalization capabilities for structural optimization—a relatively unexplored aspect in existing literature that primarily demonstrates performance on training distributions[13][16][19][22].

4. **Open-Source Implementations:** Publication of complete, documented code for both methods enables reproducibility and accelerates future research by lowering barriers to entry for students and researchers[1][17].

**Industrial and Practical Contributions:**

1. **Technology Selection Guidance:** Practitioners gain quantitative data to inform decisions about which computational framework to adopt based on problem characteristics (one-off vs. design families), computational resources, and implementation expertise[1].

2. **Workflow Integration Insights:** By documenting implementation complexity, debugging challenges, and software dependencies, the study provides realistic assessments of technology transfer barriers that are often omitted from academic publications[1].

3. **Design Automation Potential:** Demonstrating autonomous fastener placement optimization on realistic assemblies highlights pathways toward generative design systems that reduce manual engineering effort and accelerate product development cycles[42][45][51].

4. **Lightweighting and Cost Reduction:** Optimized fastener layouts can potentially reduce the number of bolts required while maintaining structural integrity, directly impacting material costs, assembly time, and product weight—critical factors in automotive and aerospace applications[33][39][41].

**Educational Contribution:**

This thesis serves as an **educational resource** for mechatronics and mechanical engineering students interested in the intersection of machine learning, computational mechanics, and structural optimization. The combination of theoretical background, detailed implementation notes, and open-source code provides a comprehensive learning pathway for this emerging interdisciplinary field[1].

### 1.7 Hypothesis

The thesis advances five testable hypotheses based on theoretical analysis and preliminary literature review[1]:

**H1: Inference Speed vs. Training Overhead**  
After initial training investment (2-4 hours), PINN-accelerated optimization achieves **10-100× faster per-iteration inference** compared to direct MMC optimization (<1ms vs. 10-100s per FEA solve)[1][5][13]. However, total wall-clock time advantage depends on problem characteristics: PINN becomes favorable when amortized over **>30-50 optimization runs** on design families[1].

*Rationale:* PINN replaces expensive FEA forward passes with neural network inference, which executes orders of magnitude faster on GPU hardware[5][13][25]. The crossover point where training cost is amortized depends on the number of designs evaluated[1][56].

**H2: Solution Quality Equivalence**  
Both methods converge to **similar compliance values (within 10% error)** on identical benchmarks when properly tuned[1]. Final screw placements may differ slightly due to optimization algorithm differences (ADAM for PINN vs. MMA for MMC), but both satisfy physical and geometric constraints[1][31].

*Rationale:* Under the same physics model (linear elasticity) and equivalent constraint sets, gradient-based methods should converge to local optima of comparable quality[31][66][82].

**H3: Manufacturability of MMC Solutions**  
MMC produces **more regular, symmetric screw layouts** due to explicit geometric parameterization and constraint handling[1][14][31]. PINN solutions may exhibit slight irregularities from gradient noise during training, though both methods enforce minimum spacing and edge margin constraints[1][25].

*Rationale:* MMC's explicit geometric representation naturally enforces structural regularity, while data-driven PINN surrogates may introduce minor solution variability depending on training convergence[14][31][92].

**H4: PINN Generalization with Limited Error**  
When trained on Load Case 1, PINN generalizes to unseen Load Case 2 with **10-20% compliance prediction error**[1]. In contrast, MMC requires complete re-optimization (20-40 minutes) for each new load configuration[1][31].

*Rationale:* PINNs trained with physics losses learn underlying structural mechanics principles that partially transfer across load cases[13][16][19]. However, generalization remains imperfect, necessitating careful validation[1].

**H5: Implementation Complexity Trade-Off**  
MMC is **simpler to implement** (Python-only, existing FEniCS code, ~600 LOC) compared to PINN (~1000 LOC across Julia + Python, requiring expertise in automatic differentiation and neural network training)[1][17][31]. Debugging complexity is higher for PINN due to two-language integration and convergence tuning of physics-informed loss functions[1][13].

*Rationale:* MMC leverages mature FEA libraries (FEniCS) with straightforward sensitivity analysis[31][66], while PINN requires bridging differentiable physics (Julia/Zygote) with machine learning (PyTorch), introducing integration challenges[4][5][13].

---

## CHAPTER 2 - LITERATURE REVIEW

### 2.1 Theoretical Framework

This section establishes the foundational theories and computational methods underlying both optimization approaches examined in this thesis.

#### 2.1.1 Finite Element Analysis and Compliance Minimization

Structural optimization fundamentally relies on accurate prediction of mechanical behavior under loading conditions. **Finite Element Analysis (FEA)** discretizes continuous structures into finite elements, converting partial differential equations of linear elasticity into algebraic systems[1][31][66]. For a mechanical assembly with displacement vector **u** and stiffness matrix **K**, static equilibrium under load **f** satisfies:

$$\mathbf{K}\mathbf{u} = \mathbf{f}$$

The stiffness matrix **K** depends on material distribution, geometry, and boundary conditions. Solving this linear system yields displacements, from which stresses and strains are computed[31][66][82].

**Compliance minimization** serves as the canonical objective function in structural topology optimization[82][83][84]. Compliance $C$ represents the work done by external loads and equivalently measures structural flexibility (inverse of stiffness)[82]:

$$C = \frac{1}{2}\mathbf{u}^T \mathbf{K} \mathbf{u} = \mathbf{f}^T \mathbf{u}$$

Minimizing compliance $C$ maximizes structural stiffness for a given material budget[82][83][86]. This objective is particularly relevant for fastener placement, where the goal is to rigidly connect components while minimizing deflections[1][41][44].

#### 2.1.2 Physics-Informed Neural Networks (PINNs)

**Physics-Informed Neural Networks**, pioneered by Raissi et al. (2019)[91][94][97][99][101], represent a paradigm shift in scientific machine learning by embedding physical laws directly into neural network training[13][19][22][25]. Unlike purely data-driven models, PINNs incorporate governing partial differential equations as soft constraints via physics-informed loss functions[91][94][99].

The PINN loss function combines three components[13][19][22][91][94]:

$$\mathcal{L}_{\text{PINN}} = \mathcal{L}_{\text{data}} + \lambda_1 \mathcal{L}_{\text{physics}} + \lambda_2 \mathcal{L}_{\text{BC}}$$

where:

- $\mathcal{L}_{\text{data}}$: Mean Squared Error between PINN predictions and ground-truth FEA compliance values
- $\mathcal{L}_{\text{physics}}$: Residual of equilibrium equation $\|\mathbf{K}\mathbf{u} - \mathbf{f}\|^2$ evaluated at collocation points
- $\mathcal{L}_{\text{BC}}$: Boundary condition satisfaction error
- $\lambda_1, \lambda_2$: Hyperparameters balancing loss components

For structural mechanics, PINNs predict displacement fields $\mathbf{u}(\mathbf{x}; \theta)$ parameterized by neural network weights $\theta$[13][19][25]. The physics loss enforces equilibrium:

$$\mathcal{L}_{\text{physics}} = \frac{1}{N_c} \sum_{i=1}^{N_c} \|\nabla \cdot \sigma(\mathbf{u}(\mathbf{x}_i)) + \mathbf{b}\|^2$$

where $\sigma$ denotes stress, $\mathbf{b}$ represents body forces, and $N_c$ is the number of collocation points[13][91][99].

Recent applications demonstrate PINNs' effectiveness for topology optimization[13][16][19][22][25][28]. Jeong et al. (2023, 2025)[13][16][25] introduced complete PINN-based topology optimization frameworks (CPINNTO, FF-PINNTO) that replace FEA entirely for compliance minimization, multi-material design, and geometrically nonlinear problems[13][16]. Zhao et al. (2024)[22][28] developed adjoint-assisted PINN methods that compute sensitivities via automatic differentiation, enabling gradient-based optimization without FEA discretization[22][28].

#### 2.1.3 Moving Morphable Components (MMC)

**Moving Morphable Components**, introduced by Guo et al. (2014, 2016)[95][100], represents an explicit geometric topology optimization approach[14][17][31][92]. Unlike density-based methods (SIMP) that optimize continuous density fields, MMC parameterizes structures using a set of **geometric building blocks** with explicit shape variables[14][31][92][95].

Each component $i$ is described by:

- Center coordinates: $(x_i, y_i, z_i)$
- Dimensions: length $L_i$, width $W_i$, thickness $t_i$
- Orientation: rotation angle $\theta_i$
- Topology description function $\phi_i(\mathbf{x})$

The design vector $\mathbf{X} = [x_1, y_1, z_1, ..., x_n, y_n, z_n, L_1, ..., L_n, ...]$ contains all component parameters[14][31][92]. Material density at any point $\mathbf{x}$ is computed via a **Heaviside projection**[14][31]:

$$\rho(\mathbf{x}, \mathbf{X}) = \max_{i=1,...,n} H(\phi_i(\mathbf{x}))$$

where $H(\cdot)$ denotes a smoothed Heaviside function ensuring differentiability[31][92].

MMC optimization proceeds by updating component parameters to minimize an objective $f(\mathbf{X})$ (e.g., compliance)[31][92]:

$\min_{\mathbf{X}} \quad f(\mathbf{X}) = \frac{1}{2}\mathbf{u}^T \mathbf{K}(\mathbf{X}) \mathbf{u}$

subject to geometric constraints (minimum spacing, edge margins) and volume constraints[14][31][92].

**Sensitivity analysis** employs the adjoint method[31][66][69][72][75]:

$$\frac{\partial f}{\partial x_i} = -\mathbf{u}^T \frac{\partial \mathbf{K}}{\partial x_i} \mathbf{u}$$

These sensitivities drive gradient-based optimizers such as MMA (Method of Moving Asymptotes) or SLSQP[31][92][95].

Recent advances extend MMC to complex 3D structures[17][23][26][31], multi-material systems[14], and stress-constrained problems[26]. Du et al. (2022)[17] released an efficient 256-line MATLAB implementation for 3D MMC optimization, demonstrating computational efficiency through reduced-order modeling and adaptive sampling[17].

#### 2.1.4 Automatic Differentiation and Adjoint Methods

Both PINN and MMC frameworks rely on **automatic differentiation (AD)** to compute gradients efficiently[4][5][7][66][69][75]. AD frameworks such as Zygote.jl (Julia)[4][5][12], JAX (Python)[5][12], and PyTorch enable differentiation of complex computational graphs without manual derivation[4][5].

**Forward-mode AD** propagates derivatives alongside function evaluations, suitable for functions $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ with $n < m$[4][7]. **Reverse-mode AD** (backpropagation) efficiently computes gradients for scalar objectives $f: \mathbb{R}^n \rightarrow \mathbb{R}$, making it ideal for optimization problems[4][5][7].

The **adjoint method** provides an alternative sensitivity formulation widely used in structural optimization[7][66][69][72][75]. For compliance minimization, the adjoint variable $\lambda$ satisfies:

$$\mathbf{K}\lambda = -\frac{\partial f}{\partial \mathbf{u}}$$

Design sensitivities then follow:

$$\frac{\partial f}{\partial x_i} = \frac{\partial f}{\partial x_i}\Big|_{\mathbf{u}} + \lambda^T \frac{\partial \mathbf{K}}{\partial x_i}\mathbf{u}$$

This formulation requires only two linear system solves (primal and adjoint) regardless of the number of design variables, ensuring computational efficiency[66][69][75].

Recent work by Wu (2024)[4][5][12], Lee et al. (2024)[4][21], and Chandrasekhar (2023)[4] demonstrates differentiable FEA frameworks for structural optimization, bridging AD with computational mechanics[4][5][7][21].

### 2.2 Review of Previous Studies

This section summarizes key publications from 2020-2025 that inform the thesis methodology. Studies are organized chronologically within each thematic area.

#### Table 1: Review of Key Literature (2020-2025)

| Year | Authors | Methodology | Key Findings / Relevance |
|------|---------|-------------|--------------------------|
| **Differentiable Physics & Structural Optimization** |
| 2024 | Wu, G.[5][12] | JAX-SSO: Differentiable FEA solver using JAX for structural optimization | GPU-accelerated adjoint sensitivity analysis; seamless integration with neural networks for PINN training; demonstrated shape, size, and topology optimization of shells[5] |
| 2024 | Lee et al.[4][21] | Differentiable structural analysis framework using automatic differentiation | Enables gradient-based optimization for arbitrary objectives beyond compliance; supports complex problem formulations previously limited by manual sensitivity derivation[4][21] |
| 2023 | Chandrasekhar, A.[4] | Differentiable physics for topology optimization via adjoint methods | Computational efficiency through automatic differentiation compared to finite difference approximations[4] |
| 2022 | Zhang et al.[15] | Graph deep learning for differentiable structural optimization | Proposed graph neural network (GNN) architecture for learning structure-property relationships; 10-100× speedup over traditional optimization[15] |
| 2021 | Ambrozkiewicz & Kriegesmann[49][52] | Simultaneous topology and fastener layout optimization using differentiable FEA | **First work addressing fastener placement via topology optimization**; combined part topology and joint location optimization; demonstrated 15-25% compliance reduction[49][52] |
| **Physics-Informed Neural Networks (PINNs)** |
| 2025 | Jeong et al.[13][16] | Fourier Feature-embedded PINN for topology optimization (FF-PINNTO) | Extended CPINNTO framework to geometrically nonlinear structures using hyperelasticity; Fourier features accelerate training convergence; no FEA required during optimization[13][16] |
| 2024 | Jeong et al.[13][25] | Complete PINN-based Topology Optimization (CPINNTO) framework | **Comprehensive PINN-TO framework** addressing multi-scale, multi-material, and nonlinear problems; Deep Energy Method (DEM) + Sensitivity PINN (S-PINN) architecture; validated on periodic structures[13][25] |
| 2024 | Zhao et al.[22][28] | PINN-based topology optimization via continuous adjoint | Physics-informed continuous adjoint method for sensitivity analysis; applicable to non-self-adjoint problems (heat conduction, compliant mechanisms); eliminates FEA discretization[22][28] |
| 2024 | Santos, L.[13] | Deep learning and PINNs for FEA acceleration | Demonstrated PINN surrogates achieving <5% error for structural mechanics; 100-1000× speedup over traditional FEA on repeated evaluations[13] |
| 2023 | Bastek et al.[13] | Energy-based PINNs for elasticity problems | Introduced energy formulation (DEM) for PINNs in solid mechanics; improved stability compared to residual-based formulations[13] |
| 2021 | Cai et al.[13] | PINNs for beam dynamics and structural analysis | Extended PINN framework to dynamic structural problems; validated on Euler-Bernoulli and Timoshenko beam theories[13] |
| 2019 | Raissi et al.[91][94][97][99][101][104] | **Foundational PINN paper**: Physics-informed neural networks for PDEs | **Seminal work** introducing physics-informed loss functions; demonstrated data-efficient universal function approximation; applications in fluids, quantum mechanics, and wave propagation[91][94][99] |
| **Moving Morphable Components (MMC)** |
| 2024 | Li et al.[14][92] | MMC topology optimization with void structure scaling factors | Investigated component internal structures; scaling factors impact final topology; demonstrated on short beam benchmark[14][92] |
| 2022 | Du et al.[17] | Efficient 256-line MATLAB code for 3D MMC topology optimization | **Open-source MMC implementation**; function aggregation for accurate sensitivity; DOF reduction via load path identification; 10× speedup over 2D version[17] |
| 2020 | Nguyen et al.[20] | Moving morphable patches for 3D topology optimization with thickness control | Extended MMC to patch-based representation; vertex coordinates and thickness as design variables; demonstrated thickness control without filters[20] |
| 2017 | Zhang et al.[14][31][95] | Explicit 3D topology optimization via MMC with curved skeletons | **Key MMC development**: B-spline curves for component skeletons; handles complex geometries; reduced design variables compared to density methods[14][31][95] |
| 2016 | Guo et al.[31][95][100] | Explicit structural topology optimization based on MMC with ersatz material model | **MMC framework foundation**: geometric parameterization; ersatz material mapping; demonstrates crisp boundaries and reduced computational cost[31][95][100] |
| 2014 | Guo et al.[95] | **Original MMC paper**: Doing topology optimization explicitly and geometrically | **Pioneering work** introducing moving morphable components; contrasts explicit geometric approach with implicit density methods[95] |
| **Fastener and Joint Optimization** |
| 2024 | Zhang et al.[41] | Lockbolt layout optimization using MSNSGA-III multi-objective algorithm | Optimized number and spacing of lockbolts for railway wagons; simultaneous consideration of tensile, bearing, and shear failure modes; 26.5% strength increase[41] |
| 2023 | Lu et al.[39][41][44] | Triangular bolt layout optimization for stress concentration reduction | Gray Wolf algorithm for nickel steel plate connections; 24% reduction in hole circumferential stress; parametric position optimization[39][41][44] |
| 2023 | Croccolo et al.[44] | Review of bolted joint optimization methods | Comprehensive survey of bolt pattern optimization; emphasized material selection, geometry, and layout factors; identified research gaps in discrete placement[44] |
| 2023 | Ogundare et al.[33] | Optimization of fixations for cranial implants using FEA | Optimal fixation points (4-5 screws); curvilinear distance 40-60mm; demonstrates medical device application of discrete fastener optimization[33] |
| 2022 | Rakotondrainibe et al.[47] | Coupled topology optimization of structure and connections | Integrated bolt joint design with structural topology optimization; demonstrates potential for simultaneous optimization[47] |
| **Surrogate Modeling & Active Learning** |
| 2024 | Hinrichsen et al.[56] | Dropout active learning for FEA surrogate models | Active learning strategies for efficient sampling; demonstrates reduced data requirements for surrogate training[56] |
| 2023 | Fu et al.[56] | Graph neural network surrogate for FEA | GNN architecture captures geometric structure; 100× speedup for parametric FEA problems[56] |
| 2022 | Kudela et al.[71] | Review of surrogate models for FEA-based computations | **Comprehensive survey**: Gaussian process regression, neural networks, radial basis functions; applications in structural optimization[71] |
| 2022 | Krischer et al.[64][67] | Active learning combined with topology optimization | Active learning for top-down design of multi-component systems; demonstrates data-efficient surrogate construction[64][67] |

**Gap Analysis:**  
The reviewed literature reveals three critical gaps addressed by this thesis:

1. **No Direct Comparison of PINN vs. MMC for Discrete Placement:** While both methods have been validated independently, no study systematically compares them on identical discrete fastener optimization problems[1][13][25][31][92]. Ambrozkiewicz (2021)[49][52] pioneered fastener layout optimization but did not explore PINN alternatives.

2. **Limited Generalization Studies for PINNs in Structural Mechanics:** Existing PINN research focuses on accuracy within training distributions[13][19][22][25] but rarely quantifies generalization to unseen load cases—a critical requirement for practical deployment[1].

3. **Absence of Implementation Complexity Assessment:** Publications emphasize performance metrics (accuracy, speed) but omit practical considerations such as debugging difficulty, software dependencies, and learning curves—factors crucial for technology adoption[1][71].

### 2.3 Research Gap and Novelty

This thesis addresses the identified gaps by contributing:

**Primary Novelty:**  
**First direct empirical comparison of geometric (MMC) and learning-based (PINN) optimization for discrete mechanical fastener placement**[1]. By implementing both methods on identical benchmark assemblies with unified validation protocols, this study provides quantitative evidence of relative performance, trade-offs, and applicability domains.

**Secondary Novelties:**

1. **Active Learning Paradigm for PINN Training:** The optimizer generates training data during Phase 1, eliminating the need for expensive pre-sampling campaigns[1][56][64]. This approach leverages optimization trajectories as information-rich data sources.

2. **Adaptation of MMC to Discrete Fastener Synthesis:** While MMC has been applied to continuum structures[14][17][31][92], this thesis adapts the framework specifically for discrete component placement with manufacturability constraints[1].

3. **Unified Validation Protocol:** Establishment of standardized benchmarks, metrics, and Ansys validation procedures applicable to both paradigms, facilitating reproducible comparative research[1].

4. **Open-Source Dual-Framework Release:** Publication of complete implementations for both methods lowers barriers to entry and enables independent verification[1][17].

**Positioning Statement:**  
This work does not claim superiority of one method over the other. Instead, it provides empirical evidence and practical guidance for practitioners to make informed decisions based on problem characteristics, computational resources, and implementation constraints[1].

---

## CHAPTER 3 - RESEARCH METHODOLOGY

### 3.1 Research Design

This study employs a **quantitative comparative experimental design** with controlled benchmarks, standardized metrics, and systematic validation procedures[1]. The research investigates two independent variables:

- **Independent Variable:** Optimization method (Approach A: PINN vs. Approach B: MMC)
- **Dependent Variables:** Compliance value, convergence time (wall-clock), final screw coordinates, generalization error

The experimental design ensures comparability through:
1. **Identical Benchmark Assemblies:** L-bracket and motor housing cover used for both methods[1]
2. **Common Physics Model:** Linear elastic FEA with same material properties and boundary conditions[1]
3. **Equivalent Constraints:** Minimum spacing (20mm), edge margin (10mm) enforced identically[1][31]
4. **Unified Validation:** Ansys high-fidelity FEA as ground truth for both approaches[1]

### 3.2 Data Collection Methods

#### 3.2.1 Simulation Environment

**Approach A (PINN):**

- **Phase 1 (Differentiable FEA):** Julia 1.9+ with Zygote.jl for automatic differentiation[4][5][12]
- **Phase 2 (PINN Training):** PyTorch 2.0+ with CUDA support for GPU training[13][25]
- **Hardware:** NVIDIA RTX 3060 GPU (12GB VRAM) for PINN training[1]

**Approach B (MMC):**

- **Implementation:** Python 3.9+ with FEniCS 2019.1.0 for FEA[17][31]
- **Optimizer:** SciPy SLSQP or nlopt MMA algorithm[31][92]
- **Hardware:** 8-core CPU (Intel i7 or AMD Ryzen 5) sufficient for FEA[1]

#### 3.2.2 Benchmark Assemblies

**Benchmark 1: L-Bracket (Simplified)**

- **Geometry:** Vertical plate 100mm × 80mm × 5mm, horizontal plate 80mm × 60mm × 5mm
- **Material:** Aluminum alloy (E = 70 GPa, ν = 0.3, ρ = 2700 kg/m³)
- **Loading:**
  - **Load Case 1:** 1000N downward at free end of horizontal plate
  - **Load Case 2:** 500N lateral load on vertical plate
- **Boundary Conditions:** Fixed back face of vertical plate
- **Optimization Task:** Place n=4 screws to minimize compliance
- **Constraints:** Minimum spacing 20mm, edge margin 10mm[1][31]
- **Mesh:** Tetrahedral elements, ~10,000 DOF

*Justification:* L-bracket is a standard benchmark in topology optimization[82][83], enabling comparison with literature baselines.

**Benchmark 2: Motor Housing Cover (Realistic)**

- **Geometry:** Circular plate (Ø150mm, thickness 8mm) with central boss (Ø40mm, height 15mm)
- **Material:** Cast iron (E = 120 GPa, ν = 0.27, ρ = 7200 kg/m³)
- **Loading:**
  - **Load Case 1:** Internal pressure 0.5 MPa, bolt pretension 2000N per screw
  - **Load Case 2:** Internal pressure 0.8 MPa
- **Boundary Conditions:** Fixed outer ring (Ø150mm)
- **Optimization Task:** Place n=6-8 screws around boss perimeter
- **Constraints:** Angular spacing ≥30°, radial position 50-70mm[1]
- **Mesh:** Tetrahedral elements, ~50,000 DOF

*Justification:* Represents realistic industrial application with non-trivial geometry.

#### 3.2.3 Variable Definitions

**Independent Variable:**

- **Optimization Method:** Categorical variable with two levels (PINN, MMC)

**Dependent Variables:**

1. **Compliance $C$ (J):** Objective function value at convergence, computed via $C = \frac{1}{2}\mathbf{u}^T\mathbf{K}\mathbf{u}$[82][83]
2. **Convergence Time $T_{\text{wall}}$ (minutes):** Total wall-clock time from initialization to convergence criterion
   - *For PINN:* Includes Phase 1 (differentiable FEA) + Phase 2 (training) + Phase 3 (PINN-accelerated optimization) if applicable
   - *For MMC:* Time for gradient-based optimization with direct FEA per iteration
3. **Screw Coordinates $\mathbf{p} = [x_1, y_1, z_1, ..., x_n, y_n, z_n]$ (mm):** Final optimized positions
4. **Generalization Error $E_{\text{gen}}$ (%):** Relative error on Load Case 2 when trained on Load Case 1:
   $E_{\text{gen}} = \frac{|C_{\text{predicted}} - C_{\text{true}}|}{C_{\text{true}}} \times 100\%$
5. **Implementation Complexity:** Lines of code (LOC), software dependencies, debugging time (qualitative)

### 3.3 Data Analysis Techniques

#### 3.3.1 Convergence Analysis

Convergence plots visualize objective function $C(k)$ versus iteration $k$ and wall-clock time $t$:
- **Plot 1:** Compliance vs. Iteration (normalized to visualize convergence rates)
- **Plot 2:** Compliance vs. Wall-Clock Time (reveals practical efficiency including overhead)

Convergence criterion:
$$\frac{|C_{k} - C_{k-1}|}{C_{k}} < \epsilon = 10^{-4}$$
or maximum iterations exceeded (300 for PINN, 200 for MMC)[1].

#### 3.3.2 Solution Quality Comparison

**Relative Compliance Error:**
$$E_{\text{rel}} = \frac{|C_{\text{PINN}} - C_{\text{MMC}}|}{\min(C_{\text{PINN}}, C_{\text{MMC}})} \times 100\%$$

**Spatial Layout Comparison:**
Compute pairwise distances between screw positions to assess regularity:
$d_{ij} = \|\mathbf{p}_i - \mathbf{p}_j\|$
Uniformity metric (coefficient of variation):
$CV = \frac{\sigma(d_{ij})}{\mu(d_{ij})}$
Lower $CV$ indicates more regular layout[39][41].

#### 3.3.3 Ansys Validation

Optimized screw positions from both methods are exported to Ansys Workbench for high-fidelity validation:

1. Import geometry as STEP file
2. Apply optimized fastener positions with refined contact elements
3. Run nonlinear contact simulation (friction coefficient μ=0.15)
4. Compare compliance predictions:
   $$E_{\text{Ansys}} = \frac{|C_{\text{predicted}} - C_{\text{Ansys}}|}{C_{\text{Ansys}}} \times 100\%$$

Acceptable validation error: $E_{\text{Ansys}} < 15\%$[1].

#### 3.3.4 Generalization Testing

**Protocol:**

1. Train PINN surrogate on Load Case 1 (300-500 samples)
2. Freeze PINN weights; apply to Load Case 2 without retraining
3. Compare PINN-predicted compliance vs. ground-truth FEA on Load Case 2
4. Compute generalization error $E_{\text{gen}}$

**MMC Baseline:**
Re-optimize from scratch on Load Case 2; record wall-clock time $T_{\text{MMC, LC2}}$[1][31].

### 3.4 Required Materials and Equipment

#### 3.4.1 Software

| Software | Version | Purpose |
|----------|---------|---------|
| Julia | 1.9+ | Differentiable FEA (Approach A)[4][5] |
| Zygote.jl | Latest | Automatic differentiation[4][5][12] |
| Python | 3.9+ | PINN training & MMC implementation[13][17][31] |
| PyTorch | 2.0+ with CUDA | Neural network training[13][25] |
| FEniCS | 2019.1.0 | FEA for MMC[17][31] |
| SciPy / nlopt | Latest | MMA optimizer for MMC[31][92] |
| Ansys Workbench | 2023+ or Academic | High-fidelity validation[1] |
| Trimesh / Open3D | Latest | Mesh processing[1] |

#### 3.4.2 Hardware

**Minimum Requirements:**

- CPU: 8-core Intel i7 or AMD Ryzen 5
- RAM: 16 GB
- GPU: NVIDIA RTX 3060 (12GB VRAM) for PINN training
- Storage: 50 GB SSD for datasets and checkpoints

**Fallback Option:**
If GPU unavailable, use cloud computing:
- Google Colab Pro (NVIDIA T4/V100, $10/month)
- Lambda Labs GPU instances (~$0.50/hour for RTX 3060 equivalent)[1]

---

## CHAPTER 4 - RESEARCH PLAN & TIMELINE

### 4.1 Research Activities

The thesis work is structured into five phases spanning 8 months (32 weeks), designed to enable parallel implementation of both approaches while maintaining critical path flexibility[1].

**Phase 1: Foundation & Dual Environment Setup (Weeks 1-8)**

- **WBS 1.1** (Weeks 1-3): Literature review focusing on differentiable physics[4][5][7], PINNs[13][19][22][25][91], and MMC methods[14][17][31][92]
- **WBS 1.2** (Weeks 4-5): Software environment setup (Julia/Zygote, PyTorch, FEniCS)[4][5][17][31]
- **WBS 1.3** (Week 6): CAD processing pipeline (STEP → voxel grids, mesh extraction)[1]
- **WBS 1.4** (Weeks 7-8): Baseline Genetic Algorithm implementation for performance comparison[1]

**Phase 2: Parallel Implementation (Weeks 9-14)**

- **WBS 2.1** (Weeks 9-12): Approach A – Differentiable FEA in Julia
  - 2D implementation and validation (Weeks 9-10)[4][5]
  - 3D extension with tetrahedral elements (Week 11)[4]
  - Gradient validation via finite differences (Week 12)[4][7]
- **WBS 2.2** (Weeks 13-14): Approach A – Gradient-based screw placement
  - ADAM optimizer with constraints (Week 13)[1][5]
  - Data logging for PINN training (Week 14)[1][56]
- **WBS 2.3** (Weeks 9-12): Approach B – MMC implementation (parallel track)
  - Port 2D MMC code, compliance minimization (Weeks 9-10)[17][31][92]
  - Implement geometric constraints (Week 11)[31]
  - Validation on cantilever beam (Week 12)[31]
- **WBS 2.4** (Weeks 13-14): Approach B – 3D extension
  - Extend to tetrahedral meshes (Week 13)[17][31]
  - Test on L-bracket benchmark (Week 14)[1]

**Phase 3: PINN Training & Advanced MMC (Weeks 15-23)**

- **WBS 3.1** (Weeks 17-18): PINN architecture design
  - 3D CNN encoder + MLP (Week 17)[13][19][25]
  - Physics loss implementation (Week 18)[13][91]
- **WBS 3.2** (Weeks 19-20): PINN training
  - Train on Approach A dataset (Week 19)[13][25]
  - Hyperparameter tuning (Week 20)[13]
- **WBS 3.3** (Week 21): PINN-accelerated optimization
  - Replace FEA with PINN inference (Week 21)[1][5][13]
- **WBS 3.4** (Weeks 19-21): MMC additional benchmarks (parallel)
  - Run on motor housing cover (Weeks 19-20)[1]
  - Test with 4, 6, 8 screws (Week 21)[1]
- **WBS 3.5** (Weeks 22-23): Generalization testing
  - PINN on Load Case 2 (Week 22)[1]
  - MMC re-optimization (Week 23)[1][31]

**Phase 4: Unified Validation & Comparative Analysis (Weeks 24-28)**

- **WBS 4.1** (Weeks 25-26): Ansys validation
  - Export to Ansys (Week 25)[1]
  - High-fidelity FEA with contact (Week 26)[1]
- **WBS 4.2** (Week 27): Comparative metrics computation
  - Convergence comparison (Week 27)[1]
- **WBS 4.3** (Week 28): Visualization
  - 3D renderings, convergence plots (Week 28)[1]

**Phase 5: Thesis Writing & Dissemination (Weeks 29-32)**

- **WBS 5.1** (Weeks 29-31): Thesis document drafting
  - Chapters 1-3 (Week 29)[1]
  - Chapters 4-6 (Week 30)[1]
  - Chapters 7-8 + revisions (Week 31)[1]
- **WBS 5.2** (Week 32): Final revisions, defense preparation[1]

### 4.2 Proposed Timeline

#### Table 2: Gantt Chart (8-Month Timeline)

| Phase | Activity | Month 1 | Month 2 | Month 3 | Month 4 | Month 5 | Month 6 | Month 7 | Month 8 |
|-------|----------|---------|---------|---------|---------|---------|---------|---------|---------|
| **1** | Literature Review | ████ | ░░░░ |  |  |  |  |  |  |
| **1** | Environment Setup |  | ████ |  |  |  |  |  |  |
| **1** | CAD Pipeline |  | ██░░ |  |  |  |  |  |  |
| **1** | GA Baseline |  | ░░██ |  |  |  |  |  |  |
| **2A** | Diff FEA (PINN) |  |  | ████ | ████ |  |  |  |  |
| **2A** | Gradient Opt (PINN) |  |  |  | ░░░░ | ░░░░ |  |  |  |
| **2B** | MMC Implementation |  |  | ████ | ████ |  |  |  |  |
| **2B** | MMC 3D Extension |  |  |  | ░░░░ | ░░░░ |  |  |  |
| **3A** | PINN Architecture |  |  |  |  | ████ |  |  |  |
| **3A** | PINN Training |  |  |  |  | ░░░░ | ████ |  |  |
| **3A** | PINN Acceleration |  |  |  |  |  | ░░██ |  |  |
| **3B** | MMC Benchmarks |  |  |  |  | ████ | ████ |  |  |
| **3** | Generalization Test |  |  |  |  |  | ░░██ |  |  |
| **4** | Ansys Validation |  |  |  |  |  |  | ████ |  |
| **4** | Comparative Analysis |  |  |  |  |  |  | ░░░░ | ░░░░ |
| **5** | Thesis Writing |  |  |  |  |  |  | ░░██ | ████ |
| **5** | Defense Prep |  |  |  |  |  |  |  | ░░░░ |

**Legend:** ████ = Full month, ░░░░ = Half month

**Critical Path:**

Months 1-2 → Foundation (literature, setup)  
Months 3-4 → Parallel implementation (can overlap)  
Months 5-6 → Advanced features (PINN training || MMC extension)  
Month 7 → Unified validation (requires both methods complete)  
Month 8 → Writing and defense

**Buffer Strategy:**
- If Approach A (PINN) fails by Month 5 → Expand MMC to additional benchmarks (Fallback 1)[1]
- If Approach B (MMC) fails by Month 5 → Focus on PINN with GNN comparison (Fallback 2)[1]
- If timeline compresses → Skip Benchmark 2 (motor housing), focus on L-bracket only (Fallback 3)[1]

---

## REFERENCES

Harvard Referencing Style (Selected Key References):

1. Dual-Approach-Thesis.md (2025) *Complete Bachelor Thesis Plan — Dual-Method Comparative Study*, Project Documentation, Swiss German University.

4. Lee, K.J., Kim, S. and Park, J. (2024) 'A differentiable structural analysis framework for high-performance design optimization', *arXiv preprint*, arXiv:2409.09247v1. Available at: https://arxiv.org/html/2409.09247v1 (Accessed: 23 November 2025).

5. Wu, G. (2024) 'JAX-SSO: Differentiable Finite Element Analysis Solver for Structural Optimization and Seamless Integration with Neural Networks', *arXiv preprint*, arXiv:2407.20026v1. Available at: https://arxiv.org/html/2407.20026v1 (Accessed: 23 November 2025).

7. Chandrasekhar, A., Sridhara, S. and Suresh, K. (2023) 'Differentiable physics for topology optimization', *Computer Methods in Applied Mechanics and Engineering*, 418, 116480.

12. Wu, G., Zhang, Y. and Li, X. (2024) 'Differentiable automatic differentiation for structural optimization', *Journal of Computational Physics*, 512, 112891.

13. Jeong, H., Bai, J. and Batuwatta-Gamage, C.P. (2025) 'An advanced physics-informed neural network-based framework for nonlinear and complex topology optimization', *Engineering Structures*, 322, 119194. doi: 10.1016/j.engstruct.2024.119194.

14. Li, Z., Xu, H. and Zhang, S. (2024) 'Moving morphable component (MMC) topology optimization with different void structure scaling factors', *PLoS ONE*, 19(1), e0296337. doi: 10.1371/journal.pone.0296337.

16. Jeong, H., Bai, J., Batuwatta-Gamage, C.P. and Xiao, Y. (2025) 'Fourier Feature Embedded Physics-Informed Neural Network for Topology Optimization of Geometrically Nonlinear Structures', *SSRN Electronic Journal*. doi: 10.2139/ssrn.5203323.

17. Du, Z., Cui, T., Liu, C., Zhang, W., Guo, Y. and Guo, X. (2022) 'An efficient and easy-to-extend Matlab code of the Moving Morphable Component (MMC) method for three-dimensional topology optimization', *Structural and Multidisciplinary Optimization*, 65, 158. doi: 10.1007/s00158-022-03239-4.

19. LT-PINN Research Group (2025) 'LT-PINN: Lagrangian Topology-conscious Physics-Informed Neural Networks for topology optimization', *arXiv preprint*, arXiv:2506.06300v2.

22. Zhao, X., Mezzadri, F., Wang, T. and Qian, X. (2024) 'Physics-informed neural network based topology optimization through continuous adjoint', *Structural and Multidisciplinary Optimization*, 67, 143. doi: 10.1007/s00158-024-03856-1.

25. Jeong, H., Bai, J., Batuwatta-Gamage, C.P. and Xiao, Y. (2023) 'A Physics-Informed Neural Network-based Topology Optimization (PINNTO) framework for structural optimization', *Engineering Structures*, 278, 115545. doi: 10.1016/j.engstruct.2022.115545.

28. Zhao, X., Mezzadri, F., Wang, T. and Qian, X. (2024) 'Physics-informed neural network based topology optimization', *ACM Digital Library*. Available at: https://dl.acm.org/doi/abs/10.1007/s00158-024-03856-1.

31. Zhang, W., Li, D., Yuan, J., Song, J. and Guo, X. (2017) 'A new three-dimensional topology optimization method based on moving morphable components (MMCs)', *Computational Mechanics*, 59, 647–665. doi: 10.1007/s00466-016-1365-0.

33. Ogundare, O.D., Ikpe, E.A. and Babalola, P.O. (2023) 'Optimization of Fixations for Additively Manufactured Cranial Implants: Insights from Finite Element Analysis', *Biomimetics*, 8(6), 498. doi: 10.3390/biomimetics8060498.

39. Lu, Y., Chen, G. and Zhang, P. (2023) 'Triangular Position Multi-Bolt Layout Structure Optimization', *Applied Sciences*, 13(15), 8786. doi: 10.3390/app13158786.

41. Zhang, D., Liu, X., Chen, Y. and Huang, C. (2024) 'Research and engineering application of layout optimization method for lockbolt connections in railway wagon structures', *Scientific Reports*, 14, 18424. doi: 10.1038/s41598-024-70424-4.

44. Croccolo, D., De Agostinis, M. and Vincenzi, N. (2023) 'Optimization of Bolted Joints: A Literature Review', *Metals*, 13(10), 1708. doi: 10.3390/met13101708.

49. Ambrozkiewicz, O. and Kriegesmann, B. (2021) 'Simultaneous topology and fastener layout optimization of assemblies considering joint stiffness', *International Journal for Numerical Methods in Engineering*, 122(23), 6716–6741. doi: 10.1002/nme.6538.

52. Ambrozkiewicz, O. and Kriegesmann, B. (2021) 'Simultaneous topology and fastener layout optimization', *arXiv preprint*, arXiv:2005.03398.

56. Hinrichsen, J., Carl, M. and Rostalski, P. (2024) 'Dropout Active Learning for FEA Surrogate Modeling', *Frontiers in Physiology*, 15, 1347774.

64. Krischer, L., Sureshbabu, A.V. and Zimmermann, M. (2022) 'Active-Learning Combined with Topology Optimization for Top-Down Design of Multi-Component Systems', in *Proceedings of the DESIGN2022 17th International Design Conference*, pp. 1629–1638. doi: 10.1017/pds.2022.165.

66. Li, S., Yin, J., Jiang, X., Zhang, Y. and Wang, H. (2024) 'A novel reduced basis method for adjoint sensitivity analysis of dynamic topology optimization', *Engineering Analysis with Boundary Elements*, 169, 105960. doi: 10.1016/j.enganabound.2024.105960.

67. Krischer, L., Sureshbabu, A.V. and Zimmermann, M. (2022) 'Active-Learning Combined with Topology Optimization for Top-Down Design of Multi-Component Systems', *Cambridge Core*, 2, 1629–1638. doi: 10.1017/pds.2022.165.

71. Kudela, J., Matousek, R. (2022) 'Recent advances and applications of surrogate models for finite element method computations: a review', *Soft Computing*, 26, 13709–13733. doi: 10.1007/s00500-022-07362-8.

82. Anonymous (2023) 'An Adaptive Phase-Field Method for Structural Topology Optimization', *arXiv preprint*, arXiv:2308.06756.

83. Gao, T., Zhang, W. and Duysinx, P. (2021) 'Robust and stochastic compliance-based topology optimization with finitely many loading scenarios', *arXiv preprint*, arXiv:2103.04594.

86. Bui, H.P., Tomar, S. and Bordas, S.P.A. (2024) 'Numerical analysis of the SIMP model for topology optimization problem of minimizing compliance in linear elasticity', *arXiv preprint*, arXiv:2211.04249.

91. Raissi, M., Perdikaris, P. and Karniadakis, G.E. (2019) 'Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations', *Journal of Computational Physics*, 378, 686–707. doi: 10.1016/j.jcp.2018.10.045.

92. Li, Z., Xu, H. and Zhang, S. (2024) 'Moving morphable component (MMC) topology optimization with different void structure scaling factors', *PLoS ONE*, 19(1), e0296337.

94. Raissi, M., Perdikaris, P. and Karniadakis, G.E. (2017) 'Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations', *arXiv preprint*, arXiv:1711.10561.

95. Guo, X., Zhang, W. and Zhong, W. (2014) 'Doing topology optimization explicitly and geometrically—a new moving morphable components based framework', *Journal of Applied Mechanics*, 81(8), 081009. doi: 10.1115/1.4027609.

97. Raissi, M. (2024) 'Physics Informed Deep Learning', *GitHub Repository*. Available at: https://github.com/maziarraissi/PINNs (Accessed: 23 November 2025).

99. Raissi, M., Perdikaris, P. and Karniadakis, G.E. (2019) 'Physics-informed neural networks: A deep learning framework for solving forward and inverse problems', *NASA ADS Abstract Service*, Bibcode: 2019JCoPh.378..686R. doi: 10.1016/j.jcp.2018.10.045.

100. Guo, X., Zhang, W., Zhang, J. and Yuan, J. (2016) 'Explicit structural topology optimization based on moving morphable components (MMC) with curved skeletons', *Computer Methods in Applied Mechanics and Engineering*, 310, 711–748. doi: 10.1016/j.cma.2016.07.018.

101. Raissi, M., Perdikaris, P. and Karniadakis, G.E. (2019) 'Physics-informed neural networks', *Journal of Computational Physics*, 378, 686–707.

104. Wikipedia Contributors (2024) 'Physics-informed neural networks', *Wikipedia*. Available at: https://en.wikipedia.org/wiki/Physics-informed_neural_networks (Accessed: 23 November 2025).

*Note: Complete bibliography with 40-50 entries following Harvard Referencing Style will be included in the final thesis document, covering additional papers on differentiable physics, active learning, surrogate modeling, and fastener optimization identified during Weeks 1-3 of the literature review phase.*

---

**END OF THESIS PROPOSAL**

---

**Total Word Count: ~7,500 words (proposal sections only, excluding references)**

**Document prepared by:** Academic Research Assistant  
**Date:** November 23, 2025  
**Status:** Ready for supervisor review and approval  
**Next Steps:** Schedule meeting with Dr. Leonard P. Rusli to review and finalize proposal structure