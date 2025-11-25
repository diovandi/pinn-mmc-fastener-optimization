<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# give me the mmc refactor, then the bigger julia generator, then the ansys export bridge

Here are all three pieces in order: an MMC refactor that matches your current script, a bigger/safer Julia data generator, and an Ansys export bridge template. [attached_file:8047c378-890c-4a6a-9f63-6946a1dd5a79][^1]

***

## 1. MMC refactor (module + runner)

This splits your current `MMC_LBracket.py` into a reusable core module and a clean benchmark script that logs iterations and compliance. It also adds a proper compliance calculation using $C = \mathbf{F}^\top \mathbf{u}$.[^1]

### `mmc_core.py`

```python
# mmc_core.py
#
# Core MMC-style L-bracket model:
# - get_ke, build_passive, update_design
# - fe_solve returns (u, F)
# - compliance(F, u) = F^T u

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# --- 1. CONFIGURATION (same as your current script) ---

nelx, nely = 40, 40        # mesh resolution
volfrac = 0.1              # currently unused, but kept for consistency
E_base = 1.0               # aluminum (normalized)
E_void = 1e-9              # void
E_bolt = 5.0               # bolt stiffness
penal = 3.0                # SIMP penalty (not yet used explicitly)

# L-bracket dimensions in element indices
void_cutoff_x = 16         # vertical leg width = 0.4 * 40
void_cutoff_y = 16         # horizontal leg height = 0.4 * 40


# --- 2. Stiffness matrix for a single element ---

def get_ke():
    E, nu = 1.0, 0.3
    k = np.array([
        1/2-nu/6,  1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
       -1/4+nu/12, -1/8-nu/8,  nu/6,       1/8-3*nu/8
    ])
    KE = E/(1-nu**2) * np.array([
        [k[^0], k[^1], k[^2], k[^3], k[^4], k[^5], k[^6], k[^7]],
        [k[^1], k[^0], k[^7], k[^6], k[^5], k[^4], k[^3], k[^2]],
        [k[^2], k[^7], k[^0], k[^5], k[^6], k[^3], k[^2], k[^1]],
        [k[^3], k[^6], k[^5], k[^0], k[^7], k[^2], k[^1], k[^2]],
        [k[^4], k[^5], k[^6], k[^7], k[^0], k[^1], k[^2], k[^3]],
        [k[^5], k[^4], k[^3], k[^2], k[^1], k[^0], k[^7], k[^6]],
        [k[^6], k[^3], k[^2], k[^1], k[^2], k[^7], k[^0], k[^5]],
        [k[^7], k[^2], k[^1], k[^2], k[^3], k[^6], k[^5], k[^0]],
    ])
    return KE


# --- 3. Passive (void) mask for the L-bracket ---

def build_passive_mask():
    """Return passive[nely, nelx] = 1 for void region."""
    passive = np.zeros((nely, nelx), dtype=int)
    for i in range(nelx):
        for j in range(nely):
            if i > void_cutoff_x and j > void_cutoff_y:
                passive[j, i] = 1
    return passive


# --- 4. Geometry projection: screw centers -> density field ---

def update_design(design_vars, radius=4.0, beta=5.0):
    """
    design_vars: [x1, y1, x2, y2, ...] in element indices
    Returns:
        x_phys[nely, nelx] in [0,1], where 0=base Al, 1=bolt
    """
    n_screws = len(design_vars) // 2
    grid_x, grid_y = np.meshgrid(np.arange(nelx), np.arange(nely))

    phi = -1.0 * np.ones((nely, nelx), dtype=float)
    for i in range(n_screws):
        cx = design_vars[2*i]
        cy = design_vars[2*i+1]
        dist = np.sqrt((grid_x - cx)**2 + (grid_y - cy)**2)
        val = radius - dist
        phi = np.maximum(phi, val)

    x_phys = 1.0 / (1.0 + np.exp(-beta * phi))
    return x_phys


# --- 5. Finite element solve for given density field ---

def fe_solve(x_phys, ke, passive):
    """
    x_phys: density field (0=Al, 1=bolt)
    passive: 1 in void, 0 elsewhere
    Returns:
        u (ndof), F (ndof), and list of fixed dofs (for debugging)
    """
    ndof = 2 * (nelx + 1) * (nely + 1)

    # Zero-out stiffness in void region
    x_eff = x_phys.copy()
    x_eff[passive == 1] = 0.0

    # Map densities to element stiffnesses
    E_elem = E_base + x_eff * (E_bolt - E_base)

    I, J, V = [], [], []

    for elx in range(nelx):
        for ely in range(nely):
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            nodes = [n1, n2, n2 + 1, n1 + 1]
            dofs = np.array([2*n for n in nodes] + [2*n+1 for n in nodes])

            k_scaled = ke * E_elem[ely, elx]
            for i in range(8):
                for j in range(8):
                    I.append(dofs[i])
                    J.append(dofs[j])
                    V.append(k_scaled[i, j])

    K = coo_matrix((V, (I, J)), shape=(ndof, ndof)).tocsc()

    # Boundary conditions: fix top edge of vertical leg
    fixed_dofs = []
    for x in range(void_cutoff_x + 1):
        n = x * (nely + 1) + nely
        fixed_dofs.extend([2*n, 2*n+1])

    # Load: downward at tip of horizontal leg
    load_node = nelx * (nely + 1) + int(void_cutoff_y / 2)
    load_dof = 2 * load_node + 1

    F = np.zeros(ndof)
    F[load_dof] = -1.0

    # Penalty method for fixed DOFs
    K_diag = K.diagonal().copy()
    K_diag[fixed_dofs] = 1e9
    K.setdiag(K_diag)

    u = spsolve(K, F)
    return u, F, fixed_dofs


def compliance(u, F):
    """Linear static compliance C = F^T u."""
    return float(F @ u)


# --- 6. Simple MMC-like optimization loop for L-bracket ---

def run_mmc_lbracket(
    n_screws=2,
    n_iters=30,
    lr=0.5,
    radius=4.0,
    beta=5.0,
    seed=1234,
):
    """
    Returns:
        history: list of dicts with keys:
            iter, compliance, x1, y1, x2, y2, ...
    """
    np.random.seed(seed)

    passive = build_passive_mask()
    ke = get_ke()

    # Initial screw centers (same as your original script)
    x = np.array([10.0, 10.0, 30.0, 5.0], dtype=float)
    assert len(x) == 2 * n_screws

    history = []

    for it in range(n_iters + 1):
        x_phys = update_design(x, radius=radius, beta=beta)
        u, F, _ = fe_solve(x_phys, ke, passive)
        C = compliance(u, F)

        rec = {"iter": it, "compliance": C}
        for i in range(n_screws):
            rec[f"x{i+1}"] = x[2*i]
            rec[f"y{i+1}"] = x[2*i+1]
        history.append(rec)

        if it == n_iters:
            break

        # Mock "gradient" that drifts screws toward the inner corner ~ (void_cutoff_x, void_cutoff_y)
        target = np.array(
            [void_cutoff_x, void_cutoff_y] * n_screws,
            dtype=float,
        )
        grad_mock = target - x
        x = x + lr * 0.05 * grad_mock  # small step

    return history
```


### `run_mmc_lbracket.py`

```python
# run_mmc_lbracket.py
#
# Thin wrapper around mmc_core.run_mmc_lbracket:
# - runs optimization
# - saves CSV + compliance plot

import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from mmc_core import run_mmc_lbracket

RESULTS_DIR = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "mmc_lbracket_log.csv")
PLOT_PATH = os.path.join(RESULTS_DIR, "mmc_lbracket_compliance.png")

def main():
    t0 = time.time()
    history = run_mmc_lbracket(n_screws=2, n_iters=30)
    elapsed = time.time() - t0

    df = pd.DataFrame(history)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved MMC log to {CSV_PATH}")
    print(f"Total time: {elapsed:.3f} s, avg per iter: {elapsed/len(df):.4f} s")

    plt.figure(figsize=(6, 4))
    plt.plot(df["iter"], df["compliance"], marker="o", color="purple")
    plt.xlabel("Iteration")
    plt.ylabel("Compliance (J)")
    plt.title("MMC Optimization Path – L-Bracket")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    print(f"Saved compliance plot to {PLOT_PATH}")

if __name__ == "__main__":
    main()
```

You can now import `mmc_core.run_mmc_lbracket` from other scripts (e.g., for joint plots against PINN).

***

## 2. Bigger / safer Julia data generator

This wraps your existing `GenerateData_Lshape.jl` logic, keeps sampling until it hits a target number of *valid* samples, and appends them to your existing CSV. You only need to expose one function in `GenerateData_Lshape.jl` that maps screw coordinates to compliance. [attached_file:8047c378-890c-4a6a-9f63-6946a1dd5a79]

### Step 1 – in `GenerateData_Lshape.jl`

Add at the bottom (or near your existing main) a pure function that evaluates compliance for a single screw vector and reuses your current solver:

```julia
# In GenerateData_Lshape.jl

function compliance_from_screws(screws::Vector{Float64})::Float64
    # screws = [x1, y1, x2, y2]
    # Reuse your existing code path:
    #  - project to L-bracket
    #  - build stiffness
    #  - solve
    #  - return compliance
    #
    # Example wiring; adapt names to your file:
    #
    # projected = enforce_lbracket_bounds(screws)
    # c, u = solve_compliance_variableE(projected, elements, forces, E, ν, fixed_dofs)
    # return c

    error("Implement compliance_from_screws() by calling your existing solver.")
end
```


### Step 2 – new file `GenerateData_Lshape_Batch.jl`

```julia
# GenerateData_Lshape_Batch.jl
#
# Robust batch generator for PINN training data.
# Uses compliance_from_screws() from GenerateData_Lshape.jl.

using Random
using Printf
using DelimitedFiles

include("GenerateData_Lshape.jl")  # brings in compliance_from_screws

const DATA_PATH       = "../common/data/pinn_training_data.csv"
const NSAMPLES_TARGET = 100   # number of additional valid samples
const MAX_TRIES       = 2000  # safety cap

# Geometry (must match ScrewPlacement / L-bracket)
const MAX_DIM = 100.0
const THICK   = 25.0
const PADDING = 5.0

function project_to_lbracket(screws::Vector{Float64})
    newscrews = copy(screws)
    @inbounds for i in 1:2:length(newscrews)
        x = newscrews[i]
        y = newscrews[i+1]

        x = clamp(x, PADDING, MAX_DIM - PADDING)
        y = clamp(y, PADDING, MAX_DIM - PADDING)

        inside_vert = (x ≤ THICK) && (y ≤ MAX_DIM)
        inside_horz = (y ≤ THICK) && (x ≤ MAX_DIM)
        in_void = !(inside_vert || inside_horz)

        if in_void
            dist_vert = abs(x - THICK)
            dist_horz = abs(y - THICK)
            if dist_vert < dist_horz
                x = THICK - PADDING
            else
                y = THICK - PADDING
            end
        end

        newscrews[i]   = x
        newscrews[i+1] = y
    end
    return newscrews
end

function main()
    Random.seed!(1234)

    existing = isfile(DATA_PATH) ? readdlm(DATA_PATH, ',', Float64) : Array{Float64}(undef, 0, 5)
    n_existing = size(existing, 1)
    @info "Existing rows in $DATA_PATH: $n_existing"

    buf = Float64[]
    n_valid = 0
    tries  = 0

    while n_valid < NSAMPLES_TARGET && tries < MAX_TRIES
        tries += 1

        screws = [
            rand(PADDING:MAX_DIM-PADDING),
            rand(PADDING:MAX_DIM-PADDING),
            rand(PADDING:MAX_DIM-PADDING),
            rand(PADDING:MAX_DIM-PADDING),
        ]
        screws = project_to_lbracket(screws)

        c = try
            compliance_from_screws(screws)
        catch e
            @warn "Skipped sample $tries due to error" exception=(e, catch_backtrace())
            continue
        end

        if !isfinite(c)
            @warn "Skipped sample $tries: non-finite compliance $c"
            continue
        end

        push!(buf, screws[^1], screws[^2], screws[^3], screws[^4], c)
        n_valid += 1
        @printf("Sample %3d: (%.1f, %.1f, %.1f, %.1f)  C=%.3f\n",
                n_valid, screws[^1], screws[^2], screws[^3], screws[^4], c)
    end

    if n_valid == 0
        @warn "No valid samples generated."
        return
    end

    new_data = reshape(buf, :, 5)
    all_data = vcat(existing, new_data)

    open(DATA_PATH, "w") do io
        writedlm(io, all_data, ',')
    end

    @info "Saved $(size(all_data, 1)) total rows to $DATA_PATH"
end

main()
```

Run this instead of your one‑shot generator whenever you want to grow the dataset.

***

## 3. Ansys export bridge (Python → APDL macro)

This script reads final layouts (e.g., from your PINN and MMC CSV logs) and writes an APDL macro that creates keypoints at screw locations on a 2D L‑bracket with dimensions in mm. You can run the macro inside Ansys Mechanical/APDL to generate geometry and run the high‑fidelity solve.

### Assumptions

- You have a CSV like `best_layouts.csv` with columns:
`method, x1, y1, x2, y2` in **mm**, consistent with your 100×100 mm L‑bracket sketch.
- You already know how you want to model screws in Ansys (e.g., keypoints used for constraint equations or small circular holes). This bridge just places the points.


### `export_to_ansys.py`

```python
# export_to_ansys.py
#
# Read best screw layouts and emit an APDL macro that:
# - builds a 2D L-bracket plate (100 x 100 mm with inner cutout)
# - creates keypoints at each screw location for each method

import pandas as pd
import os

INPUT_CSV = "../results/best_layouts.csv"
MACRO_PATH = "../results/lbracket_screws_from_python.mac"

# Geometry in mm (match your thesis figure)
OUTER_W = 100.0
OUTER_H = 100.0
INNER_W = 75.0        # inner void width (adjust to match CAD)
INNER_H = 75.0        # inner void height
THICK   = 10.0        # plate thickness for 3D model; 0 for 2D plane stress


def main():
    df = pd.read_csv(INPUT_CSV)

    lines = []

    lines.append("! --- Auto-generated L-bracket + screw macro ---")
    lines.append("/PREP7")
    lines.append("! Units: mm, N")
    lines.append("")

    # 1) Create outer rectangle area (0,0) to (100,100)
    lines += [
        "! Outer L-bracket area",
        "K,1,0,0,0",
        "K,2,{:.3f},0,0".format(OUTER_W),
        "K,3,{:.3f},{:.3f},0".format(OUTER_W, OUTER_H),
        "K,4,0,{:.3f},0".format(OUTER_H),
        "A,1,2,3,4",
        "",
    ]

    # 2) Subtract inner void (simple square notch); adjust to match your figure
    lines += [
        "! Inner cutout / void",
        "K,11,{:.3f},0,0".format(OUTER_W - INNER_W),
        "K,12,{:.3f},{:.3f},0".format(OUTER_W - INNER_W, INNER_H),
        "K,13,0,{:.3f},0".format(INNER_H),
        "A,11,12,13",
        "ASBA,1,2",          # subtract area 2 from area 1
        "",
    ]

    kp_id = 100  # start id for screw keypoints

    for row in df.itertuples(index=False):
        method = row.method
        coords = [v for v in row[1:] if pd.notna(v)]
        assert len(coords) % 2 == 0, "Odd number of coordinates"

        lines.append(f"! --- Screw layout: {method} ---")
        nscrews = len(coords) // 2
        for i in range(nscrews):
            x = coords[2*i]
            y = coords[2*i + 1]
            lines.append("K,{:d},{:.3f},{:.3f},0".format(kp_id, x, y))
            kp_id += 1
        lines.append("")

    # Optionally mesh here (2D plane stress example)
    lines += [
        "! Element type and mesh (plane stress example)",
        "ET,1,PLANE182",
        "KEYOPT,1,3,2      ! plane stress with thickness",
        "R,1,{:.3f}".format(THICK),
        "ESIZE,5",
        "AMESH,ALL",
        "",
        "! Apply BCs & loads here or in another macro",
        "FINISH"
    ]

    os.makedirs(os.path.dirname(MACRO_PATH), exist_ok=True)
    with open(MACRO_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote APDL macro to {MACRO_PATH}")
    print("In Ansys Mechanical APDL, use:  /INPUT,'lbracket_screws_from_python','mac'")


if __name__ == "__main__":
    main()
```

You can generate `best_layouts.csv` from Python or Julia by writing out the final screw coordinates for (at minimum) the PINN‑optimal and MMC‑optimal designs, then run this script to create a repeatable Ansys macro for validation runs.

<div align="center">⁂</div>

[^1]: MMC_LBracket.py

