import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import time

# --- 1. CONFIGURATION ---
nelx, nely = 40, 40  # Mesh resolution (keep small for testing)
volfrac = 0.1        # Volume fraction of screws
E_base = 1.0         # Stiffness of Aluminum (Normalized)
E_void = 1e-9        # Stiffness of Empty Space
E_bolt = 5.0         # Stiffness of Bolt (Relative to Aluminum)
penal = 3.0          # SIMP penalty (standard)

# L-Bracket Geometry Definitions (Normalized 0-1)
# Vertical leg width = 0.4, Horizontal leg height = 0.4
void_cutoff_x = 16   # 40 * 0.4 = 16
void_cutoff_y = 16

# --- 2. FINITE ELEMENT ANALYSIS ---
def FE(x, ke, passive):
    ndof = 2 * (nelx + 1) * (nely + 1)
    
    # Apply Passive Domain (Void)
    # If an element is in the top-right void, its stiffness is E_void
    x_phys = x.copy()
    x_phys[passive == 1] = 0.001 # Weak material in void
    
    # Element Stiffness Scaling
    # E = E_min + x^p * (E_0 - E_min)
    # Here we map: 0 -> Aluminum, 1 -> Bolt (Inverse of standard TO)
    # Standard TO: 0=Void, 1=Solid.
    # OUR CASE: Background is Solid (Aluminum). Components are Bolts (Stiffer).
    # So we act essentially as multi-material optimization.
    
    E_elem = E_base + x_phys * (E_bolt - E_base)
    
    # Assemble Global Stiffness K
    I, J, V = [], [], []
    for elx in range(nelx):
        for ely in range(nely):
            e_idx = elx * nely + ely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            nodes = [n1, n2, n2 + 1, n1 + 1]
            dofs = np.array([2*n for n in nodes] + [2*n+1 for n in nodes]).flatten()
            
            # Add contribution
            # (Simplified assembly for readability - normally vectorized)
            k_scaled = ke * E_elem[ely, elx]
            
            # Dirty slow loop for prototype (will vectorize later)
            for i in range(8):
                for j in range(8):
                    I.append(dofs[i])
                    J.append(dofs[j])
                    V.append(k_scaled[i, j])
                    
    K = coo_matrix((V, (I, J)), shape=(ndof, ndof)).tocsc()
    
    # Loads & BCs
    # Fix Top Edge of Vertical Leg (Top Left)
    fixed_dofs = []
    for x in range(void_cutoff_x + 1): # 0 to 16
        # Node at (x, nely) -> Top edge
        n = x * (nely + 1) + nely # Top row? No, FEniCS numbering is different.
        # Let's assume standard rectangular grid numbering
        # Top-Left is usually Fixed in L-bracket
        # Fix X and Y
        # Node index: x * (nely+1) + y
        n = x * (nely + 1) + (nely) 
        fixed_dofs.extend([2*n, 2*n+1])

    # Load at Tip of Horizontal Leg (Bottom Right)
    # Node at (nelx, y=some_middle)
    load_node = nelx * (nely + 1) + int(void_cutoff_y / 2)
    load_dof = 2 * load_node + 1 # Y direction
    F = np.zeros(ndof)
    F[load_dof] = -1.0
    
    # Solve
    # Mask Fixed DOFs (Penalty method or slicing)
    K_diag = K.diagonal()
    K_diag[fixed_dofs] = 1e9
    K.setdiag(K_diag)
    
    # We modify F in place for BCs? No, just solve.
    u = spsolve(K, F)
    return u

# --- 3. PRE-CALC STIFFNESS MATRIX (Plane Stress Quad) ---
def get_ke():
    E, nu = 1.0, 0.3
    k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
    [k[2], k[7], k[0], k[5], k[6], k[3], k[2], k[1]],
    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[2]],
    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
    [k[6], k[3], k[2], k[1], k[2], k[7], k[0], k[5]],
    [k[7], k[2], k[1], k[2], k[3], k[6], k[5], k[0]] ])
    return KE

# --- 4. MMC GEOMETRY PROJECTION ---
def update_design(design_vars):
    # design_vars: [x1, y1, x2, y2, ...] (Centers only)
    # Fixed Radius
    R = 4.0 
    n_screws = len(design_vars) // 2
    
    grid_x, grid_y = np.meshgrid(np.arange(nelx), np.arange(nely))
    
    # Initialize Density Field (0 = Base Material)
    phi = -1.0 * np.ones((nely, nelx))
    
    for i in range(n_screws):
        cx = design_vars[2*i]
        cy = design_vars[2*i+1]
        
        # Level Set Function for Circle: R - sqrt((x-cx)^2 + ...)
        # Using simple distance for now
        dist = np.sqrt((grid_x - cx)**2 + (grid_y - cy)**2)
        
        # Union of shapes (Max)
        # If inside circle (dist < R), val > 0
        val = R - dist
        phi = np.maximum(phi, val)
        
    # Heaviside Projection (Smooth)
    # x = 1 if phi > 0 (Screw), x = 0 if phi < 0 (Base)
    beta = 5.0 # Sharpness
    x_phys = 1 / (1 + np.exp(-beta * phi))
    
    return x_phys

# --- 5. MAIN LOOP ---
def main():
    print("=== MMC Discrete Fastener Optimization ===")
    
    # 1. Define Void Region (L-Bracket shape)
    # 1 = Void, 0 = Material
    passive = np.zeros((nely, nelx))
    for i in range(nelx):
        for j in range(nely):
            # Void is top right: x > 16 AND y > 16
            if i > void_cutoff_x and j > void_cutoff_y:
                passive[j, i] = 1
                
    ke = get_ke()
    
    # 2. Initialize Screws (2 Screws)
    # Start them in the "safe" zones
    # [x1, y1, x2, y2]
    x = np.array([10.0, 10.0, 30.0, 5.0]) 
    
    print(f"Initial Position: {x}")
    
    # Optimization Loop (Simple Gradient Descent for Prototype)
    # Real MMC uses MMA, but let's verify physics first
    lr = 2.0
    
    for loop in range(20):
        x_phys = update_design(x)
        
        # Solve FEA
        u = FE(x_phys, ke, passive)
        
        # Calculate Compliance
        # C = U^T K U
        # We can approximate using element strain energy
        # This is the "Expensive" part (FEA)
        
        # Simplified Compliance Calculation (Global)
        # Note: Calculating full compliance needs reconstruction of K
        # For speed in this test script, we skip exact value printing
        # and focus on whether it runs.
        
        # Sensitivity Analysis (Finite Difference for simplicity here)
        # Real MMC uses adjoint, but implementing full adjoint for MMC 
        # requires mapping gradients through the Heaviside function.
        # Let's just check if the geometry updates correctly.
        
        # Perturb x slightly to simulate an "update"
        # In a real run, MMA would drive this.
        # Here we just drift them towards the corner to simulate "optimization"
        # (Target is roughly 16, 16)
        
        grad_mock = np.array([16.0, 16.0, 16.0, 16.0]) - x
        x = x + 0.1 * grad_mock
        
        if loop % 5 == 0:
            print(f"Iter {loop}: Screws moved to {x.round(2)}")
            
            # Save visualization
            plt.figure()
            # Show material + screws
            # Flip UD for correct orientation
            plt.imshow(1 - x_phys - passive, cmap='gray', origin='lower')
            plt.title(f"MMC Iter {loop}")
            plt.savefig(f"MMC_step_{loop}.png")
            plt.close()
            
    print("✅ MMC Prototype Finished. Check .png files.")

if __name__ == "__main__":
    main()
