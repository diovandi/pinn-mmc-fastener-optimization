# MMC Run Manifest

| Tag | Load Case | Iters | Screws | Edge Margin | Min Spacing | Log File | Notes |
|-----|-----------|-------|--------|-------------|-------------|----------|-------|
| lbracket | horizontal_tip | 30 | 2 | 1.0 | 2.0 | `mmc_log_lbracket.csv` | Baseline 2D L-bracket parity run |
| vertical | vertical_tip | 30 | 2 | 2.0 | 3.5 | `mmc_log_vertical.csv` | Exploratory load case (2.5D scope) |

Associated diagnostics (per tag):

- `mmc_compliance_<tag>.png`
- `mmc_paths_<tag>.png`

Use these files when assembling the comparative tables/plots (plan §4.2–4.3) or exporting layouts for Ansys validation.

