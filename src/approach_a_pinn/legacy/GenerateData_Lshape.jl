using LinearAlgebra
using Zygote
using SparseArrays
using Printf
using JSON
using DelimitedFiles

# --- 1. GEOMETRY CONSTRAINTS ---
function enforce_lbracket_bounds(coords)
    # L-Bracket Dimensions (must match generate_lbracket.py)
    THICK = 25.0
    MAX_DIM = 100.0
    PADDING = 5.0 # Keep screws slightly away from edge
    
    n_screws = Int(length(coords) / 2)
    new_coords = copy(coords)
    
    for i in 1:n_screws
        idx_x = 2*i-1
        idx_y = 2*i
        
        x = new_coords[idx_x]
        y = new_coords[idx_y]
        
        # 1. Clamp to outer box first [0, 100]
        x = clamp(x, PADDING, MAX_DIM - PADDING)
        y = clamp(y, PADDING, MAX_DIM - PADDING)
        
        # 2. Check if inside the "Void" (Top Right)
        # Void starts at x > THICK and y > THICK
        if x > THICK && y > THICK
            # Push to nearest valid leg
            dist_to_vert = x - THICK
            dist_to_horiz = y - THICK
            
            if dist_to_vert < dist_to_horiz
                # Closer to vertical leg -> snap x to THICK
                x = THICK - PADDING
            else
                # Closer to horizontal leg -> snap y to THICK
                y = THICK - PADDING
            end
        end
        
        new_coords[idx_x] = x
        new_coords[idx_y] = y
    end
    
    return new_coords
end

# --- 2. PHYSICS KERNEL ---
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

function element_routine_cst(n1, n2, n3, E, nu, thickness)
    b1 = n2[2] - n3[2]; b2 = n3[2] - n1[2]; b3 = n1[2] - n2[2]
    c1 = n3[1] - n2[1]; c2 = n1[1] - n3[1]; c3 = n2[1] - n1[1]
    DoubleArea = (n1[1]*(n2[2] - n3[2]) + n2[1]*(n3[2] - n1[2]) + n3[1]*(n1[2] - n2[2]))
    Area = 0.5 * max(abs(DoubleArea), 1e-6)
    factor = E / ((1 - nu^2) + 1e-6)
    B = [b1 0 b2 0 b3 0; 0 c1 0 c2 0 c3; c1 b1 c2 b2 c3 b3] .* (1.0/(2*Area))
    D = [1 nu 0; nu 1 0; 0 0 (1-nu)/2] .* factor
    return (Transpose(B) * D * B) .* (Area * thickness)
end

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

function solve_compliance(flat_coords, elements, E_field, nu, thick, loads, fixed_dofs)
    n_dofs = length(flat_coords)
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
    penalty = 1e9
    K_pen = Zygote.ignore() do
        sparse(fixed_dofs, fixed_dofs, penalty, n_dofs, n_dofs)
    end
    K_dense = Array(K + K_pen) + Diagonal(fill(1e-6, n_dofs))
    u = K_dense \ loads
    return dot(loads, u)
end

# --- 3. DATA FACTORY ---
function generate_dataset()
    println("=== Starting Data Factory (L-Shape Constrained) ===")
    
    json_path = joinpath(@__DIR__, "../../../data/cad/lbracket.json")
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
    
    output_file = joinpath(@__DIR__, "../../../data/results/pinn_training_data.csv")
    
    N_SAMPLES = 50 
    dataset = zeros(N_SAMPLES, 5)
    
    for i in 1:N_SAMPLES
        # 1. Smart Initialization
        # Initialize EITHER in vertical leg OR horizontal leg to ensure validity
        screws = zeros(4)
        for s in 1:2
            if rand() > 0.5
                # Vertical Leg: x=[0, 25], y=[0, 100]
                screws[2*s-1] = rand() * 25.0
                screws[2*s] = rand() * 100.0
            else
                # Horizontal Leg: x=[0, 100], y=[0, 25]
                screws[2*s-1] = rand() * 100.0
                screws[2*s] = rand() * 25.0
            end
        end
        
        # Enforce bounds immediately
        screws = enforce_lbracket_bounds(screws)
        
        try
            for step in 1:10
                function loss(s)
                    Ef = project_material_properties(s, centroids, E_base, E_bolt, bolt_radius)
                    return solve_compliance(coords, elements, Ef, nu, thick, loads, fixed_dofs)
                end
                
                val, grads = Zygote.withgradient(loss, screws)
                
                if isnan(val) || any(isnan, grads[1]); error("NaN"); end
                
                g = clamp.(grads[1], -50.0, 50.0)
                
                # Update
                screws .-= 2.0 .* g ./ (norm(g) + 1e-8)
                
                # --- CRITICAL: Project back to L-Shape ---
                screws = enforce_lbracket_bounds(screws)
                
                if step == 10
                    dataset[i, 1:4] = screws
                    dataset[i, 5] = val
                    @printf("Sample %d: Screws=[%.1f, %.1f], C=%.2f\n", i, screws[1], screws[2], val)
                end
            end
        catch e
            println("⚠️ Skipped sample $i: $e")
            dataset[i, :] .= 0.0
        end
    end
    
    valid_data = dataset[dataset[:, 5] .> 0, :]
    writedlm(output_file, valid_data, ',')
    println("✅ Data saved to $output_file ($(size(valid_data, 1)) valid samples)")
end

generate_dataset()
