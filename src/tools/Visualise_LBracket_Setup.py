import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_lbracket_visualization():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # --- 1. Define Geometry (L-Bracket) ---
    # Dimensions based on your thesis spec
    total_h = 100
    total_w = 100
    thick = 25
    
    # Vertices (Counter-clockwise from origin)
    vertices = [
        (0, 0),              # P1: Origin
        (total_w, 0),        # P2: Bottom Right
        (total_w, thick),    # P3: Tip Top-Edge
        (thick, thick),      # P4: Inner Corner
        (thick, total_h),    # P5: Top Right-Edge
        (0, total_h),        # P6: Top Left
        (0, 0)               # Close loop
    ]
    
    # Draw the Bracket
    poly = patches.Polygon(vertices, closed=True, facecolor='#d9d9d9', edgecolor='black', linewidth=2, label='Aluminum (70 GPa)')
    ax.add_patch(poly)
    
    # --- 2. Boundary Conditions (Fixed Support) ---
    # Top edge of vertical leg: (0, 100) to (25, 100)
    # Draw "ground" hatching lines
    support_x_start, support_x_end = 0, thick
    support_y = total_h
    
    ax.plot([support_x_start, support_x_end], [support_y, support_y], color='red', linewidth=4, label='Fixed Support (u=0)')
    
    # Add classic engineering "fixed" hash marks
    for i in range(0, thick + 5, 5):
        ax.plot([i, i-2], [support_y, support_y + 3], color='red', linewidth=1)

    # --- 3. Loads (Force) ---
    # 1000N Downward at the tip of the horizontal leg
    # Tip face is x=100, y from 0 to 25. Midpoint is y=12.5
    load_x = total_w
    load_y = thick / 2.0
    force_mag = 20 # Length of arrow for visualization
    
    # Arrow pointing DOWN
    ax.arrow(load_x, load_y + force_mag, 0, -force_mag, 
             head_width=3, head_length=5, fc='blue', ec='blue', linewidth=3, 
             length_includes_head=True, label='Load (1000 N)')
    
    ax.text(load_x + 2, load_y + 5, 'F = 1000 N', color='blue', fontsize=12, fontweight='bold')

    # --- 4. Void / Invalid Zone ---
    # Dashed box for the empty space
    void_rect = patches.Rectangle((thick, thick), total_w - thick, total_h - thick, 
                                  linewidth=1, edgecolor='gray', facecolor='none', linestyle='--')
    ax.add_patch(void_rect)
    ax.text(thick + (total_w-thick)/2, thick + (total_h-thick)/2, 'VOID\n(No Material)', 
            color='gray', ha='center', va='center', fontsize=10, style='italic')

    # --- 5. Dimensions & Annotations ---
    # Helper function for dimension lines
    def draw_dim(x1, y1, x2, y2, text, offset=5):
        ax.annotate("", xy=(x1, y1), xytext=(x2, y2), arrowprops=dict(arrowstyle='<->', lw=1))
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x - offset, mid_y, text, ha='center', va='center', backgroundcolor='white', fontsize=9)

    # Height Dimension
    ax.annotate("", xy=(-5, 0), xytext=(-5, total_h), arrowprops=dict(arrowstyle='<->'))
    ax.text(-10, total_h/2, "100 mm", va='center', rotation=90)
    
    # Width Dimension
    ax.annotate("", xy=(0, -5), xytext=(total_w, -5), arrowprops=dict(arrowstyle='<->'))
    ax.text(total_w/2, -10, "100 mm", ha='center')
    
    # Thickness Dimensions
    ax.annotate("", xy=(105, 0), xytext=(105, thick), arrowprops=dict(arrowstyle='<->'))
    ax.text(115, thick/2, "25 mm", va='center', rotation=90)

    # --- Formatting ---
    ax.set_xlim(-20, 130)
    ax.set_ylim(-20, 120)
    ax.set_aspect('equal')
    ax.set_title('L-Bracket Benchmark Setup\n(2D Plane Stress)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Create a custom legend
    from matplotlib.lines import Line2D
    custom_lines = [patches.Patch(facecolor='#d9d9d9', edgecolor='black'),
                    Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='blue', lw=2, marker='^')]
    ax.legend(custom_lines, ['Aluminum Part', 'Fixed Support', 'Applied Load'], loc='upper right')

    # Save
    plt.savefig('L_Bracket_Setup_Visual.png', dpi=300, bbox_inches='tight')
    print("✅ Diagram generated: L_Bracket_Setup_Visual.png")
    # plt.show() # Uncomment if running locally with a screen

if __name__ == "__main__":
    draw_lbracket_visualization()
