#!/usr/bin/env python3
"""
Launch differentiable FEA rollouts for every geometry described in
`dataset_plan.yaml`. This script reads the plan, constructs the correct
Julia commands, and optionally executes them sequentially.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover - guidance for user
    raise SystemExit(
        "PyYAML is required. Install it via `conda run -n pinn_env pip install pyyaml` "
        "or equivalent."
    ) from exc

SCRIPT_DIR = Path(__file__).resolve().parent
PLAN_PATH = SCRIPT_DIR.parent / "dataset_plan.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
ROLL_OUT_DIR = SCRIPT_DIR


def load_plan() -> dict:
    with PLAN_PATH.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def output_path(plan_defaults: dict, geom_name: str, load_case: str) -> Path:
    rel_dir = Path(plan_defaults["output_dir"])
    out_dir = (PROJECT_ROOT / rel_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{geom_name}_{load_case}.csv"
    return out_dir / filename


def build_command(geom: dict, dataset: dict, defaults: dict) -> list[str]:
    geom_name = geom["name"]
    load_case = dataset.get("load_case", "default")
    samples = dataset.get("samples", defaults["samples_per_load_case"])
    output = output_path(defaults, geom_name, load_case)

    if geom_name == "l_bracket":
        script = ROLL_OUT_DIR / "generate_lbracket_dataset.jl"
        cmd = [
            "julia",
            str(script),
            "--load-case",
            load_case,
            "--samples",
            str(samples),
            "--output",
            str(output),
        ]
    elif geom_name == "tapered_plate":
        script = ROLL_OUT_DIR / "generate_tapered_dataset.jl"
        cmd = [
            "julia",
            str(script),
            "--samples",
            str(samples),
            "--output",
            str(output),
        ]
    elif geom_name == "ribbed_channel":
        script = ROLL_OUT_DIR / "generate_channel_dataset.jl"
        cmd = [
            "julia",
            str(script),
            "--load-case",
            load_case,
            "--samples",
            str(samples),
            "--output",
            str(output),
        ]
    else:
        raise ValueError(f"Unsupported geometry '{geom_name}' in dataset plan.")

    return cmd


def run_commands(commands: list[list[str]], dry_run: bool) -> None:
    for cmd in commands:
        print(" ".join(cmd))
        if dry_run:
            continue
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute multi-geometry rollouts.")
    parser.add_argument(
        "--geometry",
        "-g",
        help="Only run a specific geometry (name as in dataset_plan.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    plan = load_plan()
    defaults = plan["defaults"]
    commands: list[list[str]] = []

    for geom in plan["geometries"]:
        if args.geometry and geom["name"] != args.geometry:
            continue
        for dataset in geom["rollout_plan"]["datasets"]:
            commands.append(build_command(geom, dataset, defaults))

    if not commands:
        print("No commands to run (geometry filter may be too restrictive).")
        return

    try:
        run_commands(commands, args.dry_run)
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()

