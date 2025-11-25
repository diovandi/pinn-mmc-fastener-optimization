using LinearAlgebra
using Zygote
using SparseArrays
using Printf
using JSON

# --- 1. UTILS ---
function get_centroids(flat_coords, elements)
    centroids = zeros(length(elements), 2)
    for (i, elem) in enumerate(elements)
        n1_x, n1_y = flat_coords[2*elem[1]-1], flat_coords[2*elem[1]]
        n2_x, n2_y = flat_coords[2*elem[2]-1], flat_coords[2*elem[2]]
        n3_x, n3_y = flat_coords[2*elem[3]-1], flat_coords[2*elem[3]]
        centroids[i, 1] = (n1_x + n2_x + n3_x) / 3.0
        centroids[i, 2] = (n1_y + n2_y + n3_y) / 3.0
    end
    return centroids
end

# --- 2. PHYSICS: Material Projection ---
function project_material_properties(screw_coords, centroids, E_base, E_bolt, radius)
    n_screws = Int(length(screw_coords) / 2)
    n_elems = size(centroids, 1)
    
    stiffness_delta_buffer = Zygote.Buffer(zeros(n_elems))
    
    for i in 1:n_screws
        sx = screw_coords[2*i-1]
        sy = screw_coords[2*i]
        
        for e in 1:n_elems
            cx, cy = centroids[e, 1], centroids[e, 2]
            dist_sq = (cx - sx)^2 + (cy - sy)^2
            
            # Safe Gaussian: Add epsilon to prevent underflow issues
            sigma = radius / 2.0
            weight = exp(-(dist_sq + 1e-6) / (2 * sigma^2))
            
            stiffness_delta_buffer[e] += (E_bolt - E_base) * weight
        end
    end
    
    stiffness_delta = copy(stiffness_delta_buffer)
    return E_base .+ stiffness_delta
end

# --- 3. PHYSICS KERNEL (CST) ---
function element_routine_cst(n1, n2, n3, E, nu, thickness)
    x1, y1 = n1[1], n1[2]
    x2, y2 = n2[1], n2[2]
    x3, y3 = n3[1], n3[2]
    
    b1 = y2 - y3; b2 = y3 - y1; b3 = y1 - y2
    c1 = x3 - x2; c2 = x1 - x3; c3 = x2 - x1
    
    DoubleArea = (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    Area = 0.5 * abs(DoubleArea) + 1e-10
    
    factor = E / (1 - nu^2)
    
    B = [b1 0 b2 0 b3 0; 0 c1 0 c2 0 c3; c1 b1 c2 b2 c3 b3] .* (1.0/DoubleArea)
    D = [1 nu 0; nu 1 0; 0 0 (1-nu)/2] .* factor
    
    return (Transpose(B) * D * B) .* (Area * thickness)
end

# --- 4. SOLVER ---
function solve_compliance_variable_E(flat_coords, elements, E_field, nu, thick, loads, fixed_dofs)
    n_nodes = Int(length(flat_coords) / 2)
    n_dofs = n_nodes * 2
    
    local_Ks = map(1:length(elements)) do i
        elem = elements[i]
        # Safety clamp for E
        E_elem = max(E_field[i], 100.0) 
        
        n1 = [flat_coords[2*elem[1]-1], flat_coords[2*elem[1]]]
        n2 = [flat_coords[2*elem[2]-1], flat_coords[2*elem[2]]]
        n3 = [flat_coords[2*elem[3]-1], flat_coords[2*elem[3]]]
        
        element_routine_cst(n1, n2, n3, E_elem, nu, thick)
    end
    
    I_idx, J_idx = Zygote.ignore() do
        I = Int[]; J = Int[]
        for elem in elements
            dofs = Int[]
            for n in elem
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
    
    K_total = K + K_pen
    K_dense = Array(K_total)
    
    # AD-Friendly Damping: Add small value to diagonal
    # We construct a diagonal vector and add it (Zygote likes this better than I*val)
    K_dense = K_dense + Diagonal(fill(1e-6, n_dofs))
    
    u = K_dense \ loads
    return dot(loads, u)
end

# --- 5. MAIN OPTIMIZER ---
function optimize_screws()
    println("=== Discrete Fastener Optimization (Differentiable) ===")
    
    json_path = joinpath(@__DIR__, "../../../data/cad/lbracket.json")
    if !isfile(json_path)
        println("❌ Error: Cannot find $json_path")
        return
    end
    
    data = JSON.parsefile(json_path)
    coords = Float64.(data["coords"])
    raw_elems = data["elements"]
    elements = [Int.(e) for e in raw_elems]
    fixed_dofs = Int.(data["fixed_dofs"])
    load_dof = Int(data["load_dof"])
    
    nu = Float64(data["nu"])
    thick = Float64(data["thick"])
    E_base = 70000.0  
    E_bolt = 200000.0 
    bolt_radius = 5.0 
    
    centroids = get_centroids(coords, elements)
    
    n_dofs = length(coords)
    loads = zeros(n_dofs)
    loads[load_dof] = -1000.0
    
    # Initial Guess: Place screws well within bounds
    screws = [20.0, 20.0, 15.0, 80.0]
    
    println("Initial Screws: ", screws)
    
    # Reduced Learning Rate for Stability
    lr = 0.1 
    epochs = 50
    
    # ADAM State
    m = zeros(length(screws))
    v = zeros(length(screws))
    beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8
    
    for i in 1:epochs
        function loss_fn(s)
            E_field = project_material_properties(s, centroids, E_base, E_bolt, bolt_radius)
            return solve_compliance_variable_E(coords, elements, E_field, nu, thick, loads, fixed_dofs)
        end
        
        val, grads = Zygote.withgradient(loss_fn, screws)
        
        if isnan(val)
            println("❌ NaN detected at iter $i. Screws: $screws")
            break
        end
        
        g = grads[1]
        
        # Clamp gradients to prevent explosions
        g = clamp.(g, -10.0, 10.0)
        
        m = beta1 .* m .+ (1 - beta1) .* g
        v = beta2 .* v .+ (1 - beta2) .* (g .^ 2)
        m_hat = m ./ (1 - beta1^i)
        v_hat = v ./ (1 - beta2^i)
        
        # Update
        screws = screws .- lr .* m_hat ./ (sqrt.(v_hat) .+ epsilon)
        screws = clamp.(screws, 0.0, 100.0)
        
        if i % 5 == 0
            @printf("Iter %d: Compliance = %.2f | S1: (%.1f, %.1f) S2: (%.1f, %.1f)\n", 
                    i, val, screws[1], screws[2], screws[3], screws[4])
        end
    end
    
    println("\n✅ Final Screw Positions:")
    println("   Screw 1: ", screws[1:2])
    println("   Screw 2: ", screws[3:4])
end

optimize_screws()
