# Note: Full solver code is omitted here for brevity; paste it manually.
# Assume your full solve_compliance and related functions are defined above.

# To keep this clean, only paste the timer section:

using LinearAlgebra
using Zygote
using SparseArrays
using Printf
using JSON

# --- Copy/Paste all functions (element_routine_cst, project_material_properties, etc.) here ---

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

# --- MAIN OPTIMIZATION LOOP (Timed) ---
function run_fea_benchmark()
    println("--- Running FEA Benchmark (10 steps) ---")
    
    # Load Setup (Same as OptimizeLBracket.jl)
    data = JSON.parsefile(joinpath(@__DIR__, "../../../data/cad/lbracket.json"))
    coords = Float64.(data["coords"])
    elements = [Int.(e) for e in data["elements"]]
    fixed_dofs = Int.(data["fixed_dofs"])
    load_dof = Int(data["load_dof"])
    nu = 0.3; thick = 5.0
    E_base = 70000.0; E_bolt = 200000.0
    bolt_radius = 5.0
    centroids = get_centroids(coords, elements)
    n_dofs = length(coords)
    loads = zeros(n_dofs); loads[load_dof] = -1000.0
    screws = [20.0, 20.0, 15.0, 80.0]
    
    # Run a few loops to ensure JIT compilation finishes
    for _ in 1:2
        c, g = Zygote.withgradient(s -> solve_compliance_variable_E(coords, elements, project_material_properties(s, centroids, E_base, E_bolt, bolt_radius), nu, thick, loads, fixed_dofs), screws)
    end
    
    # Start Timer
    N_ITERATIONS = 10 
    start_time = time()
    
    for i in 1:N_ITERATIONS
        # 1. Run Forward + Backward pass (the expensive step)
        c, g = Zygote.withgradient(s -> solve_compliance_variable_E(coords, elements, project_material_properties(s, centroids, E_base, E_bolt, bolt_radius), nu, thick, loads, fixed_dofs), screws)
        
        # 2. Update screws slightly (to match the PINN workload)
        screws .-= 0.000001 .* g[1]
    end
    
    total_time = time() - start_time
    time_per_iteration = total_time / N_ITERATIONS
    
    println("-" * 40)
    println(f"✅ FEA Total Time: {total_time:.4f} seconds")
    println(f"✅ FEA Time per Iteration: {time_per_iteration:.4f} seconds/iter")
    println("-" * 40)

end

# --- Assume all supporting functions (element_routine_cst, solve_compliance_variable_E, etc.) are defined above ---

run_fea_benchmark()
