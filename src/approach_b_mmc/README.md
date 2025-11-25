# MMC Benchmark Runner

The refactored MMC benchmark now lives in `mmc_core.py` (core routines) and
`run_mmc_lbracket.py` (CLI wrapper). Key features:

- Parameterized domain (`--nelx`, `--nely`, `--void-x`, `--void-y`)
- Constraint hooks for edge margins and minimum spacing
- Load-case switch (`horizontal_tip` / `vertical_tip`)
- Reproducible seeds and screw counts
- Diagnostics: compliance curve + screw trajectory plot per run

## Usage

```bash
# Default L-bracket run
conda run -n mmc_env python3 src/approach_b_mmc/run_mmc_lbracket.py --tag lbracket

# Alternate load case (scoped extension)
conda run -n mmc_env python3 src/approach_b_mmc/run_mmc_lbracket.py \
  --tag vertical --load-case vertical_tip --void-x 12 --void-y 20 \
  --edge-margin 2.0 --min-spacing 3.5
```

Outputs live in `data/results/`:

- `mmc_log_<tag>.csv` — iteration log (compliance, screw coordinates, timing, constraints)
- `mmc_compliance_<tag>.png` — compliance vs. iteration
- `mmc_paths_<tag>.png` — screw coordinate trajectories

Use the CSVs when building the comparative tables/plots described in the thesis plan.

