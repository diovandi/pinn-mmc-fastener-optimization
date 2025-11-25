using LinearAlgebra
using Zygote
using SparseArrays
using Printf

# --- 1. Physics Kernel: 2D Constant Strain Triangle (CST) ---
# Input: 3 nodes (x1,y1), (x2,y2), (x3,y3)
# Output: 6x6 Stiffness Matrix
function element_routine_cst(n1, n2, n3, E, nu, thickness)
    # 1. Shape Function Derivatives (B Matrix)
    # Area = 0.5 * det([1 x1 y1; 1 x2 y2; 1 x3 y3])
    # But Zygote hates det() on small arrays sometimes, so we do cross product manually
    x1, y1 = n1[1], n1[2]
    x2, y2 = n2[1], n2[2]
    x3, y3 = n3[1], n3[2]
    
    b1 = y2 - y3; b2 = y3 - y1; b3 = y1 - y2
    c1 = x3 - x2; c2 = x1 - x3; c3 = x2 - x1
    
    DoubleArea = (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    Area = 0.5 * abs(DoubleArea) + 1e-10
    
    # B Matrix (Strain-Displacement) - 3x6
    # [ b1  0 b2  0 b3  0 ]
    # [  0 c1  0 c2  0 c3 ]
    # [ c1 b1 c2 b2 c3 b3 ]
    # We construct B_transposed * D * B manually for efficiency/AD stability
    
    # 2. Material Matrix (D) - Plane Stress
    # [ 1  nu  0 ]
    # [ nu 1   0 ]
    # [ 0  0 (1-nu)/2 ]
    factor = E / (1 - nu^2)
    
    # 3. Element Stiffness: K = integral(B' D B) dV = B' D B * Area * thickness
    # Instead of full matrix mult, we build it directly.
    # (Simplified for readability - efficient implementations use pre-calc B)
    
    B = [
        b1 0.0 b2 0.0 b3 0.0;
        0.0 c1 0.0 c2 0.0 c3;
        c1 b1 c2 b2 c3 b3
    ] .* (1.0 / DoubleArea)
    
    D = [
        1.0 nu 0.0;
        nu 1.0 0.0;
        0.0 0.0 (1.0-nu)/2.0
    ] .* factor
    
    K_loc = (Transpose(B) * D * B) .* (Area * thickness)
    
    return K_loc
end

# --- 2. Differentiable Solver ---
function solve_2d_compliance(flat_coords, elements, E, nu, thick, loads, fixed_dofs)
    n_nodes = Int(length(flat_coords) / 2)
    n_dofs = n_nodes * 2
    
    # 1. Compute Local Stiffness (Map)
    local_Ks = map(elements) do elem
        # Element is (n1_idx, n2_idx, n3_idx)
        i1, i2, i3 = elem[1], elem[2], elem[3]
        
        p1 = [flat_coords[2*i1-1], flat_coords[2*i1]]
        p2 = [flat_coords[2*i2-1], flat_coords[2*i2]]
        p3 = [flat_coords[2*i3-1], flat_coords[2*i3]]
        
        element_routine_cst(p1, p2, p3, E, nu, thick)
    end
    
    # 2. Assemble Indices (Ignore Gradient)
    I_idx, J_idx = Zygote.ignore() do
        I = Int[]; J = Int[]
        for elem in elements
            # Map 3 nodes -> 6 DOFs
            nodes = [elem[1], elem[2], elem[3]]
            dofs = Int[]
            for n in nodes
                push!(dofs, 2*n-1) # x
                push!(dofs, 2*n)   # y
            end
            
            # 6x6 block
            for r in 1:6
                for c in 1:6
                    push!(I, dofs[r])
                    push!(J, dofs[c])
                end
            end
        end
        return I, J
    end
    
    # 3. Flatten Values
    V_vals = vcat([vec(k) for k in local_Ks]...)
    
    # 4. Global K
    K = sparse(I_idx, J_idx, V_vals, n_dofs, n_dofs)
    
    # 5. BCs (Penalty)
    penalty = 1e9
    K_pen = Zygote.ignore() do
        sparse(fixed_dofs, fixed_dofs, penalty, n_dofs, n_dofs)
    end
    
    # 6. Solve
    K_total = Array(K + K_pen) # Dense for Zygote stability
    u = K_total \ loads
    
    return dot(loads, u)
end

# --- 3. Test on a "Wall" (2 Triangles) ---
function main()
    println("=== Differentiable 2D Continuum Solver (CST) ===")
    
    # A simple 1x1 square made of 2 triangles
    # 4 -- 3
    # |  / |
    # 1 -- 2
    coords = [
        0.0, 0.0, # N1 (Bottom Left) - Fixed
        1.0, 0.0, # N2 (Bottom Right) - Fixed
        1.0, 1.0, # N3 (Top Right) - Load Here
        0.0, 1.0  # N4 (Top Left)
    ]
    
    # Connectivity (Triangle 1: 1-2-3, Triangle 2: 1-3-4)
    elems = [
        (1, 2, 3),
        (1, 3, 4)
    ]
    
    E = 200000.0 # Steel (MPa)
    nu = 0.3     # Poisson
    thick = 1.0
    
    # Load: Pull Node 3 to the Right (+X)
    loads = zeros(8)
    loads[2*3-1] = 1000.0 
    
    # Fix Bottom Nodes (1 and 2) fully
    fixed = [1, 2, 3, 4]
    
    println("1. Initial Compliance...")
    c = solve_2d_compliance(coords, elems, E, nu, thick, loads, fixed)
    @printf("   Compliance: %.4f\n", c)
    
    println("\n2. Gradient Calculation...")
    grads = Zygote.gradient(x -> solve_2d_compliance(x, elems, E, nu, thick, loads, fixed), coords)[1]
    
    # Node 4 is the only free node that doesn't have a load.
    # Let's see where it wants to move to minimize compliance.
    gx_4 = grads[7]
    gy_4 = grads[8]
    @printf("   Node 4 Gradient: (%.2f, %.2f)\n", gx_4, gy_4)
    
    if abs(gx_4) > 0.0 || abs(gy_4) > 0.0
        println("✅ SUCCESS: Continuum physics engine is differentiable!")
    else
        println("❌ FAILURE: Zero gradients.")
    end
end

main()
