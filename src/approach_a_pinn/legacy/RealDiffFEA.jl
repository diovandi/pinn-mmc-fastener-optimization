using LinearAlgebra
using Zygote
using Printf
using SparseArrays

# --- 1. Physics Kernel: Truss Element Stiffness ---
function element_routine(node1, node2, E, A)
    diff = node2 - node1
    L = norm(diff) + 1e-10
    c = diff[1] / L
    s = diff[2] / L

    k_val = (E * A) / L
    
    # Construct the 4x4 local matrix without mutation
    # Zygote handles array construction fine
    k_local = k_val .* [
         c^2   c*s  -c^2  -c*s;
         c*s   s^2  -c*s  -s^2;
        -c^2  -c*s   c^2   c*s;
        -c*s  -s^2   c*s   s^2
    ]
    
    return k_local
end

# --- 2. Differentiable Solver (Sparse Assembly) ---
function solve_truss_compliance(flat_coords, connectivity, E, A, loads, fixed_dofs)
    n_nodes = Int(length(flat_coords) / 2)
    n_dofs = n_nodes * 2
    
    # --- 1. Compute Local Stiffness Matrices (Differentiable) ---
    local_Ks = map(connectivity) do elem
        idx1, idx2 = elem[1], elem[2]
        n1 = [flat_coords[2*idx1-1], flat_coords[2*idx1]]
        n2 = [flat_coords[2*idx2-1], flat_coords[2*idx2]]
        element_routine(n1, n2, E, A)
    end
    
    # --- 2. Create Global Indices (Non-Differentiable) ---
    # We use Zygote.ignore because constructing indices doesn't affect gradients
    I_idx, J_idx = Zygote.ignore() do
        I = Int[]
        J = Int[]
        for elem in connectivity
            idx1, idx2 = elem[1], elem[2]
            dofs = [2*idx1-1, 2*idx1, 2*idx2-1, 2*idx2]
            for r in 1:4
                for c in 1:4
                    push!(I, dofs[r])
                    push!(J, dofs[c])
                end
            end
        end
        return I, J
    end
    
    # --- 3. Flatten Values (Differentiable) ---
    V_vals = vcat([vec(k) for k in local_Ks]...)
    
    # --- 4. Construct Sparse Matrix ---
    K = sparse(I_idx, J_idx, V_vals, n_dofs, n_dofs)
    
    # --- 5. Boundary Conditions ---
    penalty = 1e9
    # Create penalty matrix (Indices are constant, so we ignore construction)
    K_penalty = Zygote.ignore() do
        sparse(fixed_dofs, fixed_dofs, penalty, n_dofs, n_dofs)
    end
    
    K_constrained = K + K_penalty
    
    # --- 6. Solve ---
    K_dense = Array(K_constrained)
    u = K_dense \ loads
    compliance = dot(loads, u)
    
    return compliance
end

# --- 3. Main Execution Block ---
function main()
    println("=== Differentiable Truss Solver (Julia/Zygote) ===")
    
    # 5-Node Truss
    initial_coords = [
        0.0, 0.0,  # Node 1
        1.0, 0.0,  # Node 2
        2.0, 0.0,  # Node 3
        0.5, 1.0,  # Node 4
        1.5, 1.0   # Node 5
    ]
    
    elements = [
        (1, 2), (2, 3), 
        (4, 5),         
        (1, 4), (2, 4), (2, 5), (3, 5)
    ]
    
    E = 1000.0
    A = 0.1
    
    n_dofs = length(initial_coords)
    loads = zeros(n_dofs)
    loads[2*2] = -100.0 # Load at Node 2 Y
    
    fixed_dofs = [1, 2, 2*3-1, 2*3] # Fix Node 1 (x,y) and 3 (x,y)
    
    println("1. Computing Initial Compliance...")
    c_init = solve_truss_compliance(initial_coords, elements, E, A, loads, fixed_dofs)
    @printf("   Compliance: %.4f\n", c_init)
    
    println("\n2. Sensitivity Analysis...")
    grads = Zygote.gradient(x -> solve_truss_compliance(x, elements, E, A, loads, fixed_dofs), initial_coords)[1]
    
    println("   Gradients (dC/dx, dC/dy):")
    for i in 1:5
        gx = grads[2*i-1]
        gy = grads[2*i]
        @printf("   Node %d: %6.3f, %6.3f\n", i, gx, gy)
    end
    
    if abs(grads[8]) > 0.0 # Check Node 4 Y-gradient
        println("\n✅ SUCCESS: Physics engine is differentiable!")
    else
        println("\n❌ FAILURE: Zero gradients.")
    end
end

main()
