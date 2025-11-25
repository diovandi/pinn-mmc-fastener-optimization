# Chapter 3: Theoretical Framework

## 3.1 Problem Formulation

### 3.1.1 Discrete Fastener Placement as Optimization

The discrete fastener placement problem can be formulated as:

$$\min_{\mathbf{x}} C(\mathbf{x}) = \frac{1}{2}\mathbf{u}^T\mathbf{K}(\mathbf{x})\mathbf{u}$$

subject to:
- Geometric constraints: $\mathbf{x} \in \Omega$ (valid placement region)
- Spacing constraints: $\|\mathbf{x}_i - \mathbf{x}_j\| \geq d_{\min}$ for all screw pairs
- Edge margin constraints: $\text{dist}(\mathbf{x}_i, \partial\Omega) \geq m_{\text{edge}}$

where $\mathbf{x} = [x_1, y_1, x_2, y_2, \ldots]$ represents screw center coordinates, $\mathbf{K}$ is the stiffness matrix, and $\mathbf{u}$ is the displacement vector from the equilibrium equation $\mathbf{K}\mathbf{u} = \mathbf{f}$.

### 3.1.2 L-Bracket Benchmark

The primary benchmark is a 2D L-bracket with:
- Dimensions: 100 mm × 100 mm with 25 mm thickness
- Material: Aluminum (E = 70 GPa, ν = 0.3)
- Fasteners: 2 screws with E = 200 GPa, radius = 5 mm
- Load: -1000 N at tip of horizontal leg
- Boundary conditions: Fixed top edge of vertical leg

## 3.2 Approach A: Differentiable Physics + PINN

### 3.2.1 Differentiable FEA

Using automatic differentiation (Zygote.jl), we compute gradients of compliance with respect to screw positions:

$$\frac{\partial C}{\partial \mathbf{x}} = \frac{\partial C}{\partial \mathbf{K}} \frac{\partial \mathbf{K}}{\partial \mathbf{x}}$$

The stiffness matrix depends on screw positions through a "soft screw" Gaussian projection:

$$E(\mathbf{r}) = E_{\text{base}} + (E_{\text{bolt}} - E_{\text{base}}) \sum_i \exp\left(-\frac{\|\mathbf{r} - \mathbf{x}_i\|^2}{2r_0^2}\right)$$

where $r_0$ is the characteristic radius of influence.

### 3.2.2 PINN Surrogate

The refreshed surrogate learns a mapping $f_\theta: \mathbb{R}^{14} \to \mathbb{R}$ from screw DOFs plus geometry/load encodings to compliance:

$$f_\theta([\mathbf{x}, \mathbf{g}, \mathbf{\ell}]) \approx C(\mathbf{x}, \mathbf{g}, \mathbf{\ell})$$

- Inputs: 6 screw DOFs + 3-geometry one-hot + 5-load one-hot.
- Architecture: 14 inputs → [96, 96] Tanh hidden layers → 1 output.
- Loss: Mean Squared Error on the 180-sample multi-geometry dataset (differentiable FEA ground truth).
- Outputs: Compliance prediction plus gradients (via PyTorch autograd) for optimization replacement.

### 3.2.3 Active Learning Paradigm

The optimizer generates training data by running gradient-based optimization:
1. Initialize random screw positions
2. Run ADAM optimizer with differentiable FEA
3. Log trajectory: (screw positions, compliance) pairs
4. Train PINN on accumulated trajectories
5. Use PINN for fast inference in future optimizations

## 3.3 Approach B: Moving Morphable Components (MMC)

### 3.3.1 Component Parameterization

Each fastener is parameterized as a circular component with:
- Center coordinates: $(c_x, c_y)$
- Fixed radius: $r$
- Level set function: $\phi(\mathbf{r}) = r - \|\mathbf{r} - \mathbf{c}\|$

### 3.3.2 Density Projection

The level set is converted to a density field via Heaviside projection:

$$\rho(\mathbf{r}) = \frac{1}{1 + \exp(-\beta \phi(\mathbf{r}))}$$

where $\beta$ controls projection sharpness.

### 3.3.3 Material Interpolation

Element stiffness is interpolated between base and bolt materials:

$$E_e = E_{\text{base}} + \rho_e (E_{\text{bolt}} - E_{\text{base}})$$

### 3.3.4 Constraint Handling

Geometric constraints are enforced explicitly:
- **Spacing:** Minimum distance between component centers
- **Edge margin:** Components must stay within valid domain
- **Bounds:** Coordinate limits based on L-bracket geometry

## 3.4 Comparison Framework

### 3.4.1 Metrics

1. **Convergence Speed:** Wall-clock time per iteration
2. **Solution Quality:** Final compliance value
3. **Generalization:** Performance on unseen load cases
4. **Implementation Complexity:** Lines of code, dependencies, debugging effort

### 3.4.2 Validation Protocol

- Export optimized layouts to Ansys for high-fidelity FEA
- Compare predicted vs. Ansys compliance (target: <15% error)
- Analyze spatial distribution of screw placements
- Document convergence histories for both methods

