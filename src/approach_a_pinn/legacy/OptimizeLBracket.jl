using LinearAlgebra
using Zygote
using SparseArrays
using Printf
using JSON

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

# --- 2. SOLVER ---
function solve_compliance(flat_coords, elements, E, nu, thick, loads, fixed_dofs)
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
    
    # Solve
    K_total = Array(K + K_pen)
    u = K_total \ loads
    return dot(loads, u)
end

# --- 3. MAIN OPTIMIZATION LOOP ---
function optimize_lbracket()
    println("=== L-Bracket Generative Optimization ===")
    
    # 1. Load Data
    json_path = joinpath(@__DIR__, "../../../data/cad/lbracket.json")
    if !isfile(json_path)
        println("❌ Error: Cannot find $json_path")
        return
    end
    
    data = JSON.parsefile(json_path)
    
    # Convert JSON data to Julia Types
    coords = Float64.(data["coords"])
    # JSON elements come as Vector{Any}, need Vector{Vector{Int}}
    raw_elems = data["elements"]
    elements = [Int.(e) for e in raw_elems]
    
    fixed_dofs = Int.(data["fixed_dofs"])
    load_dof = Int(data["load_dof"])
    E = Float64(data["E"])
    nu = Float64(data["nu"])
    thick = Float64(data["thick"])
    
    n_dofs = length(coords)
    loads = zeros(n_dofs)
    loads[load_dof] = -1000.0 # Downward load
    
    println("Nodes: ", length(coords)/2)
    println("Elements: ", length(elements))
    println("Fixed DOFs: ", length(fixed_dofs))
    
    # 2. Setup Optimization Mask
    # We want to move EVERYTHING except:
    # - Fixed Nodes (Support)
    # - Load Node (Point of application)
    mask = ones(n_dofs)
    for dof in fixed_dofs
        mask[dof] = 0.0
    end
    mask[load_dof] = 0.0
    mask[load_dof-1] = 0.0 # Fix X of load node too, just to be safe
    
    # 3. Run Optimization (ADAM)
    lr = 0.5 # Higher learning rate for mm scale
    epochs = 50
    
    m = zeros(n_dofs)
    v = zeros(n_dofs)
    beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8
    
    c_initial = solve_compliance(coords, elements, E, nu, thick, loads, fixed_dofs)
    println("Initial Compliance: ", c_initial)
    
    for i in 1:epochs
        loss, grads = Zygote.withgradient(c -> solve_compliance(c, elements, E, nu, thick, loads, fixed_dofs), coords)
        g = grads[1] .* mask
        
        m = beta1 .* m .+ (1 - beta1) .* g
        v = beta2 .* v .+ (1 - beta2) .* (g .^ 2)
        m_hat = m ./ (1 - beta1^i)
        v_hat = v ./ (1 - beta2^i)
        
        # Update
        coords = coords .- lr .* m_hat ./ (sqrt.(v_hat) .+ epsilon)
        
        if i % 10 == 0
            @printf("Iter %d: Compliance = %.4f\n", i, loss)
        end
    end
    
    println("Final Compliance: ", solve_compliance(coords, elements, E, nu, thick, loads, fixed_dofs))
    println("✅ Optimization Complete.")
    
    # Optional: Save result back to JSON for viewing?
    # (You can implement this if you want to see the morphed shape in Python/Gmsh)
end

optimize_lbracket()
