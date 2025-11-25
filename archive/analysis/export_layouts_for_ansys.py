import json
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT_DIR / "data" / "results"
OUTPUT_PATH = RESULT_DIR / "ansys_payloads.json"


def load_diff_layout():
    path = RESULT_DIR / "lbracket_diff_fea_log.csv"
    df = pd.read_csv(path)
    last = df.iloc[-1]
    return {
        "method": "diff_fea",
        "tag": "lbracket",
        "compliance": float(last["compliance"]),
        "screws": [
            {"x": float(last["s1_x"]), "y": float(last["s1_y"])},
            {"x": float(last["s2_x"]), "y": float(last["s2_y"])},
        ],
        "source": path.name,
    }


def load_mmc_layout(tag):
    path = RESULT_DIR / f"mmc_log_{tag}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    last = df.iloc[-1]
    screws = []
    idx = 1
    while f"x{idx}" in last:
        screws.append({"x": float(last[f"x{idx}"]), "y": float(last[f"y{idx}"])})
        idx += 1
    return {
        "method": "mmc",
        "tag": tag,
        "compliance": float(last["compliance"]),
        "screws": screws,
        "edge_margin": float(last.get("edge_margin", 0.0)),
        "min_spacing": float(last.get("min_spacing", 0.0)),
        "source": path.name,
    }


def main():
    payloads = []
    payloads.append(load_diff_layout())
    for tag in ("lbracket", "vertical"):
        layout = load_mmc_layout(tag)
        if layout:
            payloads.append(layout)

    with open(OUTPUT_PATH, "w") as fp:
        json.dump({"layouts": payloads}, fp, indent=2)
    print(f"Exported payloads to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

