<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# why is 3d mmc out of scope?

3D MMC is “out of scope” in the sense that, relative to your current state and thesis goals, the cost‑to‑benefit ratio is terrible.

## 1. Where MMC stands now

- You already have MMC working and benchmarked on three **2D** geometries: L‑bracket, tapered plate, ribbed channel, with logs, trajectories, and convergence plots.[^1]
- Even in 2D, MMC shows tricky numerical behavior (huge normalized compliances and spikes on the ribbed‑channel loads), which you still need to explain and, ideally, renormalize.[^1]

So MMC has already fulfilled its role in the thesis: a geometric, direct‑optimization baseline that you can compare against your multi‑geometry PINN in terms of speed, robustness, and solution quality.[^1]

## 2. What 3D MMC would actually entail

To go to 3D you would need, at minimum:

- A **3D FE kernel** (Hex/Tet elements, DOF indexing, BCs, loads) wired into your MMC density/ersatz model, plus a stable 3D solver. Your current FE stack for MMC is 2D only.[^1]
- A 3D **component parameterization and projection** (cylindrical “screw” volumes, distance fields in 3D, 3D Heaviside/filtering), and updated constraint handling in 3D space.[^2]
- Sensitivities / gradients for 3D fields (either via adjoints or auto‑diff inside FEniCS/another framework), which is considerably more painful to debug than your current 2D SciPy implementation.[^2]

That is easily a multi‑month coding and debugging project on its own, with a very high risk of getting stuck on stability issues similar to (but worse than) what you already see in 2D.[^1]

## 3. How it fits (or not) with your thesis objectives

Your thesis questions are about **comparative behavior**: speed, solution quality, generalization, and implementation complexity of a learning‑based surrogate vs an explicit geometric optimizer for discrete fasteners.[^2]

- Those comparisons are already answered convincingly in 2D across multiple geometries, and the results are summarized in your comprehensive table and unified plots.[^1]
- Adding **one 3D case with PINN only** extends the “can this surrogate handle a realistic 3D part?” question, without changing the qualitative MMC conclusions. MMC being 2D‑only is itself an interesting point in the “implementation complexity and scalability” discussion.[^2][^1]

In short: 3D MMC would cost you a lot of time to reproduce *the same story* you already have in 2D, whereas 3D PINN gives you new information (3D generalization, 3D speedup vs 3D FEA) for much less marginal effort.

If your supervisor insists on 3D MMC as a requirement, that is a different conversation; but based on your current plan and results, it is rational to declare it out of scope and focus your remaining time on 3D PINN + Ansys + writing.

<div align="center">⁂</div>

[^1]: comprehensive_results_table.md

[^2]: Dual-Approach-Thesis.md

