using LinearAlgebra
using Zygote
using Printf
using SparseArrays
using DelimitedFiles # To save data for the PINN later

# --- INCLUDE YOUR SOLVER ---
# (We paste the solver functions here to keep it self-contained)

function element_routine(node1, node2, E, A)
    diff = node2 - node1
    L = norm(diff) + 1e-10
    c = diff[1] / L
    s = diff[2] / L
    k_val = (E * A) / L
    k_local = k_val .* [
         c^2   c*s  -c^2  -c*s;
         c*s   s^2  -c*s  -s^2;
        -c^2  -c*s   c^2   c*s;
        -c*s  -s^2   c*s   s^2
    ]
    return k_local
end

function solve_truss_compliance(flat_coords, connectivity, E, A, loads, fixed_dofs)
    n_nodes = Int(length(flat_coords) / 2)
    n_dofs = n_nodes * 2
    
    local_Ks = map(connectivity) do elem
        idx1, idx2 = elem[1], elem[2]
        n1 = [flat_coords[2*idx1-1], flat_coords[2*idx1]]
        n2 = [flat_coords[2*idx2-1], flat_coords[2*idx2]]
        element_routine(n1, n2, E, A)
    end
    
    I_idx, J_idx = Zygote.ignore() do
        I = Int[]; J = Int[]
        for elem in connectivity
            idx1, idx2 = elem[1], elem[2]
            dofs = [2*idx1-1, 2*idx1, 2*idx2-1, 2*idx2]
            for r in 1:4
                for c in 1:4
                    push!(I, dofs[r]); push!(J, dofs[c])
                end
            end
        end
        return I, J
    end
    
    V_vals = vcat([vec(k) for k in local_Ks]...)
    K = sparse(I_idx, J_idx, V_vals, n_dofs, n_dofs)
    
    penalty = 1e9
    K_penalty = Zygote.ignore() do
        sparse(fixed_dofs, fixed_dofs, penalty, n_dofs, n_dofs)
    end
    
    K_constrained = K + K_penalty
    
    # Zygote friendly solve
    K_dense = Array(K_constrained)
    u = K_dense \ loads
    return dot(loads, u)
end

# --- OPTIMIZER (ADAM) ---
function optimize_structure()
    println("=== Starting Generative Optimization (ADAM) ===")
    
    # Initial Setup
    coords = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.5, 1.0, 1.5, 1.0]
    elements = [(1,2), (2,3), (4,5), (1,4), (2,4), (2,5), (3,5)]
    E = 1000.0; A = 0.1
    loads = zeros(10); loads[4] = -100.0
    fixed_dofs = [1, 2, 5, 6] # Fix Node 1 and 3
    
    # Optimization parameters
    lr = 0.01  # Learning rate
    epochs = 200
    
    # Mask: We only want to move Node 4 and Node 5
    # 1 = move, 0 = stay
    # Nodes: 1(fix), 2(load-fix x?), 3(fix), 4(move), 5(move)
    # Let's allow Node 2 (Load) to move vertically, and 4, 5 to move freely
    # Indices: [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
    mask = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    
    # ADAM State
    m = zeros(length(coords))
    v = zeros(length(coords))
    beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8
    
    # Data Logging (For PINN Training Later)
    history = []
    
    for i in 1:epochs
        # 1. Compute Gradient
        loss, grads = Zygote.withgradient(c -> solve_truss_compliance(c, elements, E, A, loads, fixed_dofs), coords)
        g = grads[1]
        
        # 2. Filter Gradient (Only move allowed nodes)
        g = g .* mask
        
        # 3. ADAM Update Step
        m = beta1 .* m .+ (1 - beta1) .* g
        v = beta2 .* v .+ (1 - beta2) .* (g .^ 2)
        m_hat = m ./ (1 - beta1^i)
        v_hat = v ./ (1 - beta2^i)
        
        # Update coordinates (Gradient Descent -> move AGAINST gradient)
        coords = coords .- lr .* m_hat ./ (sqrt.(v_hat) .+ epsilon)
        
        # Log progress
        if i % 10 == 0
            @printf("Iter %d: Compliance = %.4f\n", i, loss)
            push!(history, (i, loss))
        end
    end
    
    println("\n=== Optimization Complete ===")
    println("Final Node 4 Position: ", coords[7:8])
    println("Final Node 5 Position: ", coords[9:10])
end

optimize_structure()
