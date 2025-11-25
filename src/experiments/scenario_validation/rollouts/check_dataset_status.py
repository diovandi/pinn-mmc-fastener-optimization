#!/usr/bin/env python3
"""Check current dataset status against targets from dataset_plan.yaml."""
import yaml
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PLAN_PATH = SCRIPT_DIR.parent / "dataset_plan.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = PROJECT_ROOT / "data/results/multi_geom_training"

def get_sample_count(csv_path):
    """Count samples in CSV (excluding header)."""
    if not csv_path.exists():
        return 0
    with csv_path.open("r") as f:
        return max(0, len(f.readlines()) - 1)

def main():
    with PLAN_PATH.open("r") as f:
        plan = yaml.safe_load(f)
    
    defaults = plan["defaults"]
    print("Dataset Status Check\n" + "="*60)
    
    total_needed = 0
    total_current = 0
    
    for geom in plan["geometries"]:
        geom_name = geom["name"]
        print(f"\n{geom_name.upper()}:")
        
        for dataset in geom["rollout_plan"]["datasets"]:
            load_case = dataset["load_case"]
            target = dataset.get("samples", defaults["samples_per_load_case"])
            
            # Map to actual filenames
            if geom_name == "l_bracket":
                if load_case == "horizontal_tip":
                    filename = "l_bracket_horizontal.csv"
                elif load_case == "vertical_tip":
                    filename = "l_bracket_vertical.csv"
                else:
                    filename = f"l_bracket_{load_case}.csv"
            elif geom_name == "tapered_plate":
                filename = "tapered_plate_combined.csv"
            elif geom_name == "ribbed_channel":
                if load_case == "upward_tip":
                    filename = "ribbed_channel_upward.csv"
                elif load_case == "lateral_shear":
                    filename = "ribbed_channel_shear.csv"
                else:
                    filename = f"ribbed_channel_{load_case}.csv"
            else:
                filename = f"{geom_name}_{load_case}.csv"
            
            csv_path = OUTPUT_ROOT / filename
            current = get_sample_count(csv_path)
            needed = max(0, target - current)
            
            status = "✅" if current >= target else "⏳"
            print(f"  {status} {load_case:20s}: {current:4d}/{target:4d} samples", end="")
            if needed > 0:
                print(f" (need {needed} more)")
            else:
                print(" (complete)")
            
            total_current += current
            total_needed += target
    
    print(f"\n{'='*60}")
    print(f"Total: {total_current}/{total_needed} samples ({100*total_current/total_needed:.1f}% complete)")
    
    if total_current >= total_needed:
        print("\n✅ All datasets meet target sample counts!")
        return 0
    else:
        print(f"\n⏳ Still need {total_needed - total_current} more samples.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

