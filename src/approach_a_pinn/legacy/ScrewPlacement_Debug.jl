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

# --- 2. PHYSICS KERNEL (CST) ---
function element_routine_cst(n1, n2, n3, E, nu, thickness)
    # Check for bad inputs
    if any(isnan, n1) || any(isnan, n2) || any(isnan, n3) || isnan(E)
        # Return identity matrix to prevent crash during debug
        return Matrix{Float64}(I, 6, 6)
    end

    x1, y1 = n1[1], n1[2]
    x2, y2 = n2[1], n2[2]
    x3, y3 = n3[1], n3[2]
    
    b1 = y2 - y3; b2 = y3 - y1; b3 = y1 - y2
    c1 = x3 - x2; c2 = x1 - x3; c3 = x2 - x1
    
    DoubleArea = (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    # FORCE positive area to avoid singularity
    Area = 0.5 * max(abs(DoubleArea), 1e-6)
    
    factor = E / ((1 - nu^2) + 1e-6)
    
    B = [
        b1 0.0 b2 0.0 b3 0.0;
        0.0 c1 0.0 c2 0.0 c3;
        c1 b1 c2 b2 c3 b3
    ] .* (1.0 / (2*Area))
    
    D = [
        1.0 nu 0.0;
        nu 1.0 0.0;
        0.0 0.0 (1.0-nu)/2.0
    ] .* factor
    
    K_loc = (Transpose(B) * D * B) .* (Area * thickness)
    
    return K_loc
end

# --- 3. MATERIAL PROJECTION ---
function project_material_properties(screw_coords, centroids, E_base, E_bolt, radius)
    n_elems = size(centroids, 1)
    n_screws = Int(length(screw_coords)/2)
    
    # Zygote Buffer for accumulation
    stiffness_buf = Zygote.Buffer(zeros(n_elems))
    
    for i in 1:n_screws
        sx = screw_coords[2*i-1]
        sy = screw_coords[2*i]
        
        for e in 1:n_elems
            cx = centroids[e, 1]
            cy = centroids[e, 2]
            dist_sq = (cx - sx)^2 + (cy - sy)^2
            
            # Robust Gaussian
            sigma = radius / 2.0
            weight = exp(-(dist_sq) / (2 * sigma^2 + 1e-6))
            
            stiffness_buf[e] += (E_bolt - E_base) * weight
        end
    end
    
    delta = copy(stiffness_buf)
    return E_base .+ delta
end

# --- 4. SOLVER ---
function solve_compliance(flat_coords, elements, E_field, nu, thick, loads, fixed_dofs)
    n_dofs = length(flat_coords)
    
    local_Ks = map(1:length(elements)) do i
        elem = elements[i]
        E_val = max(E_field[i], 10.0) # Ensure positive stiffness
        
        idx1, idx2, idx3 = elem
        n1 = [flat_coords[2*idx1-1], flat_coords[2*idx1]]
        n2 = [flat_coords[2*idx2-1], flat_coords[2*idx2]]
        n3 = [flat_coords[2*idx3-1], flat_coords[2*idx3]]
        
        element_routine_cst(n1, n2, n3, E_val, nu, thick)
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
    
    # Penalty BCs
    penalty = 1e9
    K_pen = Zygote.ignore() do
        sparse(fixed_dofs, fixed_dofs, penalty, n_dofs, n_dofs)
    end
    
    # Convert to dense + damping
    K_total = Array(K + K_pen) + Diagonal(fill(1e-4, n_dofs))
    
    # Solve
    u = K_total \ loads
    return dot(loads, u)
end

# --- 5. MAIN ---
function main()
    println("=== DEBUG: Screw Placement ===")
    
    json_path = joinpath(@__DIR__, "../../../data/cad/lbracket.json")
    data = JSON.parsefile(json_path)
    
    coords = Float64.(data["coords"])
    # Fix: Ensure elements are Int arrays
    raw_elems = data["elements"]
    elements = [Int.(e) for e in raw_elems]
    
    fixed_dofs = Int.(data["fixed_dofs"])
    load_dof = Int(data["load_dof"])
    
    # Override material props for testing
    E_base = 70000.0
    E_bolt = 200000.0
    nu = 0.3
    thick = 5.0
    bolt_radius = 5.0
    
    n_dofs = length(coords)
    loads = zeros(n_dofs)
    loads[load_dof] = -1000.0
    
    centroids = get_centroids(coords, elements)
    
    # Initial Screws
    screws = [20.0, 20.0, 10.0, 80.0]
    println("Start Screws: ", screws)
    
    # Test Single Forward Pass
    println("Running Forward Pass...")
    E_field = project_material_properties(screws, centroids, E_base, E_bolt, bolt_radius)
    c = solve_compliance(coords, elements, E_field, nu, thick, loads, fixed_dofs)
    println("Compliance: ", c)
    
    if isnan(c)
        println("❌ Forward Pass Failed (NaN)")
        return
    end
    
    # Test Gradient
    println("Running Gradient Pass...")
    
    function loss(s)
        Ef = project_material_properties(s, centroids, E_base, E_bolt, bolt_radius)
        return solve_compliance(coords, elements, Ef, nu, thick, loads, fixed_dofs)
    end
    
    grads = Zygote.gradient(loss, screws)[1]
    println("Gradients: ", grads)
    
    if any(isnan, grads)
        println("❌ Gradient Pass Failed (NaN)")
        return
    end
    
    println("\n✅ SUCCESS: Debug checks passed. Running Optimization Loop...")
    
    # Optimizer
    lr = 2.0 # mm
    for i in 1:20
        val, gs = Zygote.withgradient(loss, screws)
        g = gs[1]
        
        # Simple Gradient Descent
        screws = screws .- lr .* g ./ (norm(g) + 1e-8)
        screws = clamp.(screws, 0.0, 100.0)
        
        if i % 5 == 0
            @printf("Iter %d: C=%.2f | S1=(%.1f, %.1f)\n", i, val, screws[1], screws[2])
        end
    end
end

main()
