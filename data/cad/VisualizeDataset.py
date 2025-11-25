import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# --- Configuration ---
DATA_PATH = "../results/pinn_training_data.csv"

def visualize():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # 1. Load Data
    # Columns: [s1_x, s1_y, s2_x, s2_y, s3_x, s3_y, s4_x, s4_y, Compliance]
    # Note: GenerateData.jl might have saved [s1x, s1y, s2x, s2y, Compliance] (4 screws flattened? No, random init was rand(4) meaning 2 screws)
    # Wait, let's check GenerateData.jl:
    # screws = rand(4) -> This is 2 screws (x1, y1, x2, y2)
    # Dataset cols: 1:4 are coords, 5 is compliance.
    
    df = pd.read_csv(DATA_PATH, header=None)
    data = df.values
    
    compliance = data[:, -1]
    screws = data[:, :-1] # All coordinate columns
    
    print(f"Loaded {len(data)} samples.")
    print(f"Compliance Range: {compliance.min():.2f} - {compliance.max():.2f}")

    # --- Plot 1: Spatial Distribution ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw L-Bracket Outline (Hardcoded dimensions from generation script)
    # (0,0) -> (100,0) -> (100,25) -> (25,25) -> (25,100) -> (0,100) -> (0,0)
    verts = [(0,0), (100,0), (100,25), (25,25), (25,100), (0,100), (0,0)]
    codes = [patches.Path.MOVETO] + [patches.Path.LINETO]*(len(verts)-1)
    path = patches.Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='#e6e6e6', lw=2, edgecolor='black')
    ax.add_patch(patch)
    
    # Plot Screws
    # We scatter plot ALL screws from ALL samples
    # Color them by the compliance of their specific sample
    # Low Compliance (Blue/Good) -> High Compliance (Red/Bad)
    
    cmap = plt.get_cmap('viridis_r') # Reversed: Yellow=Low(Good), Purple=High(Bad) usually, let's check
    # Standard Viridis: Yellow=High, Purple=Low. 
    # We want Low Compliance to be "Good" (highlighted). 
    # Let's use 'plasma_r' (Bright=Low Compliance)
    
    sc = None
    for i in range(len(data)):
        # Extract pairs (x,y)
        # Assuming 2 screws per sample (4 columns)
        sample_screws = screws[i].reshape(-1, 2)
        c_val = compliance[i]
        
        sc = ax.scatter(sample_screws[:, 0], sample_screws[:, 1], 
                        c=[c_val]*len(sample_screws), 
                        cmap='turbo', 
                        vmin=compliance.min(), vmax=compliance.max(),
                        s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

    plt.colorbar(sc, label='Compliance (Lower is Stiffer)')
    plt.title(f"Generated Data Distribution ({len(data)} Samples)\nAre the screws clustering in high-strain areas?")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)
    
    output_img = "../results/dataset_visualization.png"
    plt.savefig(output_img, dpi=150)
    print(f"Saved spatial plot to {output_img}")
    
    # --- Plot 2: Histogram ---
    plt.figure(figsize=(8, 5))
    plt.hist(compliance, bins=15, color='skyblue', edgecolor='black')
    plt.title("Distribution of Compliance Values")
    plt.xlabel("Compliance (J)")
    plt.ylabel("Count")
    
    output_hist = "../results/dataset_histogram.png"
    plt.savefig(output_hist)
    print(f"Saved histogram to {output_hist}")

if __name__ == "__main__":
    visualize()
