using LinearAlgebra
using Zygote
using SparseArrays
using Printf
using JSON
using DelimitedFiles

# --- 1. UTILS ---
function get_centroids(flat_coords, elements)
    centroids = zeros(length(elements), 2)
    for (i, elem) in enumerate(elements)
        n1 = flat_coords[(2*elem[1]-1):(2*elem[1])]
        n2 = flat_coords[(2*elem[2]-1):(2*elem[2])]
        n3 = flat_coords[(2*elem[3]-1):(2*elem[3])]
        centroids[i, :] = (n1 + n2 + n3) / 3.0
    end
    return centroids
end

# --- 2. PHYSICS KERNEL ---
function element_routine_cst(n1, n2, n3, E, nu, thickness)
    b1 = n2[2] - n3[2]; b2 = n3[2] - n1[2]; b3 = n1[2] - n2[2]
    c1 = n3[1] - n2[1]; c2 = n1[1] - n3[1]; c3 = n2[1] - n1[1]
    
    DoubleArea = (n1[1]*(n2[2] - n3[2]) + n2[1]*(n3[2] - n1[2]) + n3[1]*(n1[2] - n2[2]))
    # Ensure area is never zero to avoid division by zero
    Area = 0.5 * max(abs(DoubleArea), 1e-6)
    
    factor = E / ((1 - nu^2) + 1e-6)
    
    B = [b1 0 b2 0 b3 0; 0 c1 0 c2 0 c3; c1 b1 c2 b2 c3 b3] .* (1.0/(2*Area))
    D = [1 nu 0; nu 1 0; 0 0 (1-nu)/2] .* factor
    
    return (Transpose(B) * D * B) .* (Area * thickness)
end

# --- 3. MATERIAL PROJECTION ---
function project_material_properties(screw_coords, centroids, E_base, E_bolt, radius)
    n_elems = size(centroids, 1)
    n_screws = Int(length(screw_coords)/2)
    
    stiffness_buf = Zygote.Buffer(zeros(n_elems))
    
    for i in 1:n_screws
        sx, sy = screw_coords[2*i-1], screw_coords[2*i]
        for e in 1:n_elems
            dist_sq = (centroids[e,1]-sx)^2 + (centroids[e,2]-sy)^2
            weight = exp(-(dist_sq) / (2 * radius^2 + 1e-6))
            stiffness_buf[e] += (E_bolt - E_base) * weight
        end
    end
    return E_base .+ copy(stiffness_buf)
end

# --- 4. SOLVER ---
function solve_compliance(flat_coords, elements, E_field, nu, thick, loads, fixed_dofs)
    n_nodes = Int(length(flat_coords) / 2)
    n_dofs = n_nodes * 2
    
    local_Ks = map(1:length(elements)) do i
        elem = elements[i]
        E_val = max(E_field[i], 100.0) 
        n1 = flat_coords[(2*elem[1]-1):(2*elem[1])]
        n2 = flat_coords[(2*elem[2]-1):(2*elem[2])]
        n3 = flat_coords[(2*elem[3]-1):(2*elem[3])]
        element_routine_cst(n1, n2, n3, E_val, nu, thick)
    end
    
    I_idx, J_idx = Zygote.ignore() do
        I = Int[]; J = Int[]
        for elem in elements
            dofs = [2*elem[1]-1, 2*elem[1], 2*elem[2]-1, 2*elem[2], 2*elem[3]-1, 2*elem[3]]
            for r in 1:6, c in 1:6
                push!(I, dofs[r]); push!(J, dofs[c])
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
    
    # Convert to dense and damp to ensure non-singularity
    K_total = Array(K + K_pen) + Diagonal(fill(1e-5, n_dofs))
    
    u = K_total \ loads
    return dot(loads, u)
end

# --- 5. DATA GENERATOR ---
function generate_dataset()
    println("=== Starting Data Factory ===")
    
    json_path = joinpath(@__DIR__, "../../../data/cad/lbracket.json")
    if !isfile(json_path)
        println("❌ Error: JSON file not found.")
        return
    end
    
    data = JSON.parsefile(json_path)
    coords = Float64.(data["coords"])
    elements = [Int.(e) for e in data["elements"]]
    fixed_dofs = Int.(data["fixed_dofs"])
    load_dof = Int(data["load_dof"])
    
    nu = 0.3; thick = 5.0
    E_base = 70000.0; E_bolt = 200000.0
    bolt_radius = 15.0 
    
    centroids = get_centroids(coords, elements)
    n_dofs = length(coords)
    loads = zeros(n_dofs)
    loads[load_dof] = -1000.0
    
    output_dir = joinpath(@__DIR__, "../../../data/results")
    if !isdir(output_dir); mkdir(output_dir); end
    output_file = joinpath(output_dir, "pinn_training_data.csv")
    
    N_SAMPLES = 50 
    dataset = zeros(N_SAMPLES, 5)
    
    println("Generating $N_SAMPLES samples...")
    
    for i in 1:N_SAMPLES
        # Safe Initialization: Keep screws away from edges initially to avoid instability
        screws = [20.0 + 60.0*rand(), 20.0 + 60.0*rand(), 
                  20.0 + 60.0*rand(), 20.0 + 60.0*rand()]
        
        # Try/Catch block for the whole optimization run
        try
            for step in 1:10
                function loss(s)
                    Ef = project_material_properties(s, centroids, E_base, E_bolt, bolt_radius)
                    return solve_compliance(coords, elements, Ef, nu, thick, loads, fixed_dofs)
                end
                
                val, grads = Zygote.withgradient(loss, screws)
                
                if isnan(val)
                    error("NaN Compliance")
                end
                
                g = grads[1]
                if any(isnan, g)
                    error("NaN Gradient")
                end
                
                # Clip Gradients
                g = clamp.(g, -50.0, 50.0)
                
                # Update
                screws .-= 2.0 .* g ./ (norm(g) + 1e-8)
                screws = clamp.(screws, 5.0, 95.0) # Keep inside box
                
                if step == 10
                    dataset[i, 1:4] = screws
                    dataset[i, 5] = val
                    @printf("Sample %d: Screws=[%.1f, %.1f], C=%.2f\n", i, screws[1], screws[2], val)
                end
            end
        catch e
            println("⚠️ Skipped sample $i due to instability: $e")
            # If failed, save zeros (we can filter later) or retry
            dataset[i, :] .= 0.0
        end
    end
    
    # Filter out failed runs (where compliance is 0)
    valid_indices = dataset[:, 5] .> 0
    final_data = dataset[valid_indices, :]
    
    writedlm(output_file, final_data, ',')
    println("✅ Data saved to $output_file ($(size(final_data, 1)) valid samples)")
end

generate_dataset()
