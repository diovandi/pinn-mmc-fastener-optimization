import gmsh
import json
import numpy as np
import os

def generate_lbracket_mesh(filename_base="lbracket"):
    print("Initializing Gmsh...")
    gmsh.initialize()
    gmsh.model.add("L_Bracket")

    # --- 1. Corrected Geometry Parameters (mm) ---
    # An L-shape with defined thickness
    Total_Height = 100.0 
    Total_Length = 100.0
    Thickness    = 25.0
    
    # Mesh Element Size (Coarse enough for speed, fine enough for physics)
    lc = 4.0 

    # --- 2. Draw Points (2D Profile) ---
    # P6 (0, 100)      P5 (25, 100)
    #  +----------------+
    #  |                |
    #  |    Vertical    |
    #  |      Leg       |
    #  |                |
    #  |                + P4 (25, 25)  Inner Corner
    #  |                |
    #  |                +---------------------+ P3 (100, 25)
    #  |                   Horizontal Leg     |
    #  +--------------------------------------+ P2 (100, 0)
    # P1 (0, 0) Origin
    
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(Total_Length, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(Total_Length, Thickness, 0, lc)
    p4 = gmsh.model.geo.addPoint(Thickness, Thickness, 0, lc)
    p5 = gmsh.model.geo.addPoint(Thickness, Total_Height, 0, lc)
    p6 = gmsh.model.geo.addPoint(0, Total_Height, 0, lc)

    # --- 3. Draw Lines ---
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p5)
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p1)

    # --- 4. Create Surface ---
    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6])
    surf = gmsh.model.geo.addPlaneSurface([loop])

    # Sync geometry
    gmsh.model.geo.synchronize()

    # --- 5. Generate Mesh ---
    gmsh.model.mesh.generate(2)
    
    # --- 6. Extract Data for Julia ---
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    
    # Reshape to Nx3
    coords_reshaped = np.array(node_coords).reshape(-1, 3)
    
    # Store as flat list [x1, y1, x2, y2...]
    flat_coords = []
    
    # Map Gmsh Tag -> Index (1-based for Julia)
    tag_to_index = {}
    
    for i, tag in enumerate(node_tags):
        tag_to_index[tag] = i + 1
        flat_coords.append(coords_reshaped[i][0])
        flat_coords.append(coords_reshaped[i][1])

    # Get Elements (Triangles only)
    # Element Type 2 = 3-node triangle
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    
    if len(elem_node_tags) == 0:
        print("Error: No 2D elements generated.")
        gmsh.finalize()
        return

    # Assuming only one type of 2D element (triangles)
    tri_node_tags = elem_node_tags[0].reshape(-1, 3)
    
    elements = []
    for tri in tri_node_tags:
        # Convert tags to local 1-based indices
        n1 = tag_to_index[tri[0]]
        n2 = tag_to_index[tri[1]]
        n3 = tag_to_index[tri[2]]
        elements.append([int(n1), int(n2), int(n3)])

    print(f"Generated Mesh: {len(node_tags)} Nodes, {len(elements)} Elements")

    # --- 7. Identify Boundary Nodes ---
    # Fixed Nodes: Top edge of Vertical Leg (y=100) 
    # (Standard L-bracket test: Fix Top, Load Tip)
    fixed_nodes = []
    for i, (x, y, z) in enumerate(coords_reshaped):
        # Fix the top edge (y approx 100)
        if abs(y - Total_Height) < 1e-3: 
            idx = i + 1
            fixed_nodes.append(2*idx - 1) # x-dof
            fixed_nodes.append(2*idx)     # y-dof
            
    # Load Node: Tip of Horizontal Leg (x=100, y=12.5)
    # Find node closest to center of tip
    target = np.array([Total_Length, Thickness/2.0, 0.0])
    dists = np.linalg.norm(coords_reshaped - target, axis=1)
    load_node_idx = np.argmin(dists) + 1
    load_dof = 2*load_node_idx # y-direction
    
    print(f"Fixed Nodes Count: {len(fixed_nodes)//2}")
    print(f"Load Node Index: {load_node_idx} (DOF: {load_dof})")

    # --- 8. Save to JSON ---
    data = {
        "coords": flat_coords,
        "elements": elements,
        "fixed_dofs": fixed_nodes,
        "load_dof": int(load_dof),
        "E": 70000.0, # Aluminum
        "nu": 0.3,
        "thick": 5.0
    }
    
    with open(f"{filename_base}.json", "w") as f:
        json.dump(data, f)
        
    # Save VTK for viewing (optional)
    gmsh.write(f"{filename_base}.vtk")
    
    gmsh.finalize()
    print(f"Saved to {filename_base}.json")

if __name__ == "__main__":
    generate_lbracket_mesh()
