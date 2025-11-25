#!/usr/bin/env python3
"""
Scale up all multi-geometry datasets to target sample counts from dataset_plan.yaml.
Uses append_to_dataset.jl to add samples to existing files.
"""

import subprocess
import sys
from pathlib import Path
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PLAN_PATH = SCRIPT_DIR.parent / "dataset_plan.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
APPEND_SCRIPT = SCRIPT_DIR / "append_to_dataset.jl"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "results" / "multi_geom_training"

def load_plan():
    with PLAN_PATH.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)

def get_current_sample_count(csv_path):
    """Count samples in existing CSV file (excluding header)."""
    if not csv_path.exists():
        return 0
    with csv_path.open("r") as f:
        lines = f.readlines()
        return max(0, len(lines) - 1)  # Subtract header

def run_append(geometry, load_case, target_samples, output_path):
    """Run the appropriate Julia append script for the geometry."""
    script_dir = SCRIPT_DIR
    if geometry == "l_bracket":
        script = script_dir / "append_lbracket.jl"
        cmd = ["julia", str(script), load_case, str(target_samples), str(output_path)]
    elif geometry == "tapered_plate":
        script = script_dir / "append_tapered.jl"
        cmd = ["julia", str(script), str(target_samples), str(output_path)]
    elif geometry == "ribbed_channel":
        script = script_dir / "append_channel.jl"
        cmd = ["julia", str(script), load_case, str(target_samples), str(output_path)]
    else:
        print(f"Unknown geometry: {geometry}")
        return False
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running append script:")
        print(result.stderr)
        return False
    print(result.stdout)
    return True

def main():
    plan = load_plan()
    defaults = plan["defaults"]
    
    tasks = []
    
    for geom in plan["geometries"]:
        geom_name = geom["name"]
        for dataset in geom["rollout_plan"]["datasets"]:
            load_case = dataset["load_case"]
            target_samples = dataset.get("samples", defaults["samples_per_load_case"])
            
            # Map geometry names to file naming (matching actual filenames)
            if geom_name == "l_bracket":
                if load_case == "horizontal_tip":
                    filename = "l_bracket_horizontal.csv"
                elif load_case == "vertical_tip":
                    filename = "l_bracket_vertical.csv"
                else:
                    filename = f"l_bracket_{load_case}.csv"
            elif geom_name == "tapered_plate":
                filename = "tapered_plate_combined.csv"
                load_case = "combined_tip"
            elif geom_name == "ribbed_channel":
                if load_case == "upward_tip":
                    filename = "ribbed_channel_upward.csv"
                elif load_case == "lateral_shear":
                    filename = "ribbed_channel_shear.csv"
                else:
                    filename = f"ribbed_channel_{load_case}.csv"
            else:
                print(f"Unknown geometry: {geom_name}")
                continue
            
            output_path = OUTPUT_ROOT / filename
            current_count = get_current_sample_count(output_path)
            
            if current_count >= target_samples:
                print(f"✅ {filename}: {current_count} samples (target: {target_samples}) - already complete")
                continue
            
            needed = target_samples - current_count
            print(f"📊 {filename}: {current_count} samples, need {needed} more (target: {target_samples})")
            tasks.append((geom_name, load_case, target_samples, output_path))
    
    if not tasks:
        print("All datasets already meet target sample counts!")
        return 0
    
    print(f"\nWill process {len(tasks)} dataset(s)...\n")
    
    for geom_name, load_case, target_samples, output_path in tasks:
        print(f"\n{'='*60}")
        print(f"Processing: {output_path.name}")
        print(f"{'='*60}")
        success = run_append(geom_name, load_case, target_samples, output_path)
        if not success:
            print(f"❌ Failed to scale up {output_path.name}")
            return 1
    
    print(f"\n✅ All datasets scaled up successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

