using LinearAlgebra
using Zygote
using SparseArrays
using Printf

# --- 1. PHYSICS KERNEL (CST Element) ---
function element_routine_cst(n1, n2, n3, E, nu, thickness)
    x1, y1 = n1[1], n1[2]
    x2, y2 = n2[1], n2[2]
    x3, y3 = n3[1], n3[2]
    
    b1 = y2 - y3; b2 = y3 - y1; b3 = y1 - y2
    c1 = x3 - x2; c2 = x1 - x3; c3 = x2 - x1
    
    DoubleArea = (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    Area = 0.5 * abs(DoubleArea) + 1e-10
    
    factor = E / (1 - nu^2)
    
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

# --- 2. SOLVER (Differentiable Assembly) ---
function solve_2d_compliance(flat_coords, elements, E, nu, thick, loads, fixed_dofs)
    n_nodes = Int(length(flat_coords) / 2)
    n_dofs = n_nodes * 2
    
    local_Ks = map(elements) do elem
        i1, i2, i3 = elem[1], elem[2], elem[3]
        p1 = [flat_coords[2*i1-1], flat_coords[2*i1]]
        p2 = [flat_coords[2*i2-1], flat_coords[2*i2]]
        p3 = [flat_coords[2*i3-1], flat_coords[2*i3]]
        element_routine_cst(p1, p2, p3, E, nu, thick)
    end
    
    I_idx, J_idx = Zygote.ignore() do
        I = Int[]; J = Int[]
        for elem in elements
            nodes = [elem[1], elem[2], elem[3]]
            dofs = Int[]
            for n in nodes
                push!(dofs, 2*n-1); push!(dofs, 2*n)
            end
            for r in 1:6
                for c in 1:6
                    push!(I, dofs[r]); push!(J, dofs[c])
                end
            end
        end
        return I, J
    end
    
    V_vals = vcat([vec(k) for k in local_Ks]...)
    K = sparse(I_idx, J_idx, V_vals, n_dofs, n_dofs)
    
    penalty = 1e9
    K_pen = Zygote.ignore() do
        sparse(fixed_dofs, fixed_dofs, penalty, n_dofs, n_dofs)
    end
    
    # Solve
    K_total = Array(K + K_pen)
    u = K_total \ loads
    return dot(loads, u)
end

# --- 3. OPTIMIZER (ADAM) ---
function optimize_continuum()
    println("=== Generative Design: 2D Continuum ===")
    
    # Setup: A 1x1 square split into 2 triangles
    # 4 -- 3 (Load ->)
    # |  / |
    # 1 -- 2 (Fixed)
    
    coords = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    elements = [(1, 2, 3), (1, 3, 4)]
    
    E = 200000.0; nu = 0.3; thick = 1.0
    
    n_dofs = 8
    loads = zeros(n_dofs)
    loads[2*3-1] = 1000.0 # X-Load at Node 3
    
    fixed_dofs = [1, 2, 3, 4] # Fix Node 1 and 2
    
    # Optimization Settings
    lr = 0.01
    epochs = 100
    
    # Mask: Only allow Node 4 (Indices 7,8) to move
    mask = zeros(8)
    mask[7] = 1.0; mask[8] = 1.0
    
    # ADAM State
    m = zeros(length(coords))
    v = zeros(length(coords))
    beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8
    
    println("Initial Compliance: ", solve_2d_compliance(coords, elements, E, nu, thick, loads, fixed_dofs))
    
    for i in 1:epochs
        loss, grads = Zygote.withgradient(c -> solve_2d_compliance(c, elements, E, nu, thick, loads, fixed_dofs), coords)
        g = grads[1] .* mask # Filter gradient
        
        # ADAM Update
        m = beta1 .* m .+ (1 - beta1) .* g
        v = beta2 .* v .+ (1 - beta2) .* (g .^ 2)
        m_hat = m ./ (1 - beta1^i)
        v_hat = v ./ (1 - beta2^i)
        
        coords = coords .- lr .* m_hat ./ (sqrt.(v_hat) .+ epsilon)
        
        if i % 20 == 0
            @printf("Iter %d: Compliance = %.4f | Node 4: (%.3f, %.3f)\n", i, loss, coords[7], coords[8])
        end
    end
    
    println("Final Node 4 Position: ", coords[7:8])
end

optimize_continuum()
