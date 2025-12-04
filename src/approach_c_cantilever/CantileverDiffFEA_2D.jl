using LinearAlgebra
using Zygote
using SparseArrays
using Printf

# Include CST element routine from existing 2D FEA
include("../approach_a_pinn/DiffFEA_2D.jl")

# ============================================================================
# Cantilever Beam Differentiable FEA - 2D Continuum (Plane Stress)
# ============================================================================

# --- Material and Geometry Constants ---
const E_2D = 210e9  # Young's modulus (Pa) - Steel
const nu_2D = 0.3   # Poisson's ratio
const b_2D = 0.1    # Beam width (m) - thickness in plane stress
const h_2D = 0.05   # Beam height (m)
const L_2D = 1.0    # Beam length (m)
const q_2D = 1000.0 # Distributed load (N/m)

# ============================================================================
# 1. Mesh Generation for 2D Rectangular Domain
# ============================================================================
function create_beam_mesh_2d(L, h, nx, ny)
    """
    Create 2D rectangular mesh [0,L] × [0,h] with triangular elements.
    nx: number of elements along length
    ny: number of elements along height
    Returns: nodes (x,y coordinates), elements (triangles), boundary_labels
    """
    # Generate nodes
    nodes = Vector{Tuple{Float64, Float64}}()
    node_map = Dict{Tuple{Int, Int}, Int}()  # (i, j) -> node_idx
    
    for j in 0:ny
        for i in 0:nx
            x = i * L / nx
            y = j * h / ny
            push!(nodes, (x, y))
            node_map[(i, j)] = length(nodes)
        end
    end
    
    # Generate triangular elements (each rectangle split into 2 triangles)
    elements = Vector{Tuple{Int, Int, Int}}()
    
    for j in 0:(ny-1)
        for i in 0:(nx-1)
            # Rectangle corners: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
            n1 = node_map[(i, j)]
            n2 = node_map[(i+1, j)]
            n3 = node_map[(i+1, j+1)]
            n4 = node_map[(i, j+1)]
            
            # Split into two triangles
            # Triangle 1: n1-n2-n3
            push!(elements, (n1, n2, n3))
            # Triangle 2: n1-n3-n4
            push!(elements, (n1, n3, n4))
        end
    end
    
    # Identify boundary nodes
    # bottom (y=0): label 1
    # right (x=L): label 2
    # top (y=h): label 3
    # left (x=0): label 4
    bottom_nodes = [node_map[(i, 0)] for i in 0:nx]
    right_nodes = [node_map[(nx, j)] for j in 0:ny]
    top_nodes = [node_map[(i, ny)] for i in 0:nx]
    left_nodes = [node_map[(0, j)] for j in 0:ny]
    
    boundary_labels = Dict(
        :bottom => bottom_nodes,
        :right => right_nodes,
        :top => top_nodes,
        :left => left_nodes
    )
    
    return nodes, elements, boundary_labels
end

# ============================================================================
# 2. Load Vector Assembly for Distributed Load on Top Edge
# ============================================================================
function assemble_distributed_load_2d(nodes, elements, boundary_labels, q, L)
    """
    Assemble load vector for distributed load q on top edge.
    q: distributed load (N/m)
    Returns: load vector F (2 DOF per node: ux, uy)
    """
    n_nodes = length(nodes)
    n_dofs = 2 * n_nodes
    F = zeros(n_dofs)
    
    # Get top edge nodes
    top_nodes = boundary_labels[:top]
    
    # Distribute load to top edge nodes
    # For each node on top edge, apply vertical force
    # Total force = q * L, distribute evenly among top nodes
    n_top = length(top_nodes)
    if n_top > 1
        # Edge nodes get half share, interior nodes get full share
        force_per_node = q * L / (n_top - 1)  # Approximate
        
        for (idx, node_idx) in enumerate(top_nodes)
            if idx == 1 || idx == n_top
                # Corner nodes: half force
                F[2*node_idx] = q * L / (2 * (n_top - 1))
            else
                # Interior nodes: full force
                F[2*node_idx] = q * L / (n_top - 1)
            end
        end
    else
        # Single node case
        F[2*top_nodes[1]] = q * L
    end
    
    return F
end

# ============================================================================
# 3. Intermediate Support Penalty (2D)
# ============================================================================
function apply_intermediate_support_penalty_2d(K, nodes, support_positions, L, penalty_base, sigma)
    """
    Apply penalty method for intermediate supports in 2D.
    Penalty is applied to vertical displacement (uy) at nodes near support position.
    """
    n_nodes = length(nodes)
    n_dofs = 2 * n_nodes
    
    # Find nodes near support positions
    I_pen = Int[]
    J_pen = Int[]
    V_pen = Float64[]
    
    for pos in support_positions
        pos_clamped = clamp(pos, 0.1*L, 0.9*L)
        
        for (node_idx, (x, y)) in enumerate(nodes)
            # Only apply to nodes in valid range
            if x >= 0.1*L && x <= 0.9*L
                dist_sq = (x - pos_clamped)^2
                weight = exp(-dist_sq / (2 * sigma^2))
                
                if weight > 1e-6  # Only apply if significant
                    # Vertical DOF is 2*node_idx
                    dof = 2*node_idx
                    push!(I_pen, dof)
                    push!(J_pen, dof)
                    push!(V_pen, penalty_base * weight)
                end
            end
        end
    end
    
    # Combine penalties (sum if multiple supports affect same DOF)
    if !isempty(I_pen)
        # Use sparse matrix to handle duplicates
        K_pen = sparse(I_pen, J_pen, V_pen, n_dofs, n_dofs)
        return K_pen
    else
        return spzeros(n_dofs, n_dofs)
    end
end

# ============================================================================
# 4. Main 2D Solver
# ============================================================================
function solve_beam_compliance_2d(support_positions::Vector{Float64};
                                   L=L_2D, h=h_2D, nx=50, ny=5,
                                   E=E_2D, nu=nu_2D, b=b_2D, q=q_2D)
    """
    Solve 2D continuum FEA for cantilever beam and compute compliance.
    support_positions: vector of support positions along beam (in [0.1L, 0.9L])
    Returns: compliance C = F' * u
    """
    # Create mesh (constant, not dependent on support positions)
    nodes, elements, boundary_labels = Zygote.ignore() do
        create_beam_mesh_2d(L, h, nx, ny)
    end
    n_nodes = length(nodes)
    n_dofs = 2 * n_nodes
    
    # Convert nodes to flat coordinate array for CST routine
    flat_coords = Float64[]
    for (x, y) in nodes
        push!(flat_coords, x)
        push!(flat_coords, y)
    end
    
    # Compute element stiffness matrices using CST
    local_Ks = map(elements) do elem
        i1, i2, i3 = elem[1], elem[2], elem[3]
        n1 = [flat_coords[2*i1-1], flat_coords[2*i1]]
        n2 = [flat_coords[2*i2-1], flat_coords[2*i2]]
        n3 = [flat_coords[2*i3-1], flat_coords[2*i3]]
        element_routine_cst(n1, n2, n3, E, nu, b)
    end
    
    # Assemble global stiffness matrix
    I_idx, J_idx = Zygote.ignore() do
        I = Int[]
        J = Int[]
        for elem in elements
            nodes_elem = [elem[1], elem[2], elem[3]]
            dofs = Int[]
            for n in nodes_elem
                push!(dofs, 2*n-1)  # ux
                push!(dofs, 2*n)    # uy
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
    
    V_vals = vcat([vec(k) for k in local_Ks]...)
    K = sparse(I_idx, J_idx, V_vals, n_dofs, n_dofs)
    
    # Assemble load vector (constant)
    F = Zygote.ignore() do
        assemble_distributed_load_2d(nodes, elements, boundary_labels, q, L)
    end
    
    # Boundary conditions: fixed at left edge (x=0)
    left_nodes = boundary_labels[:left]
    fixed_dofs = Int[]
    for node_idx in left_nodes
        push!(fixed_dofs, 2*node_idx - 1)  # ux = 0
        push!(fixed_dofs, 2*node_idx)      # uy = 0
    end
    
    # Apply fixed boundary conditions via penalty
    penalty_base = 1e9
    K_pen_fixed = Zygote.ignore() do
        sparse(fixed_dofs, fixed_dofs, penalty_base, n_dofs, n_dofs)
    end
    
    # Apply intermediate support penalty (differentiable)
    sigma = L / (2 * nx)  # Characteristic length scale
    K_pen_support = apply_intermediate_support_penalty_2d(
        K, nodes, support_positions, L, penalty_base, sigma
    )
    
    # Solve
    K_total = Array(K + K_pen_fixed + K_pen_support)  # Dense for Zygote stability
    u = K_total \ F
    
    # Compliance
    compliance = dot(F, u)
    
    return compliance
end

# ============================================================================
# 5. Extended Solver with Displacement Output
# ============================================================================
function solve_beam_compliance_extended_2d(support_positions::Vector{Float64};
                                           L=L_2D, h=h_2D, nx=50, ny=5,
                                           E=E_2D, nu=nu_2D, b=b_2D, q=q_2D)
    """
    Solve 2D continuum FEA and return compliance, displacement, and nodes.
    Used for extracting tip deflection and other post-processing.
    """
    # Create mesh
    nodes, elements, boundary_labels = Zygote.ignore() do
        create_beam_mesh_2d(L, h, nx, ny)
    end
    n_nodes = length(nodes)
    n_dofs = 2 * n_nodes
    
    # Convert nodes to flat coordinate array
    flat_coords = Float64[]
    for (x, y) in nodes
        push!(flat_coords, x)
        push!(flat_coords, y)
    end
    
    # Compute element stiffness matrices
    local_Ks = map(elements) do elem
        i1, i2, i3 = elem[1], elem[2], elem[3]
        n1 = [flat_coords[2*i1-1], flat_coords[2*i1]]
        n2 = [flat_coords[2*i2-1], flat_coords[2*i2]]
        n3 = [flat_coords[2*i3-1], flat_coords[2*i3]]
        element_routine_cst(n1, n2, n3, E, nu, b)
    end
    
    # Assemble global stiffness
    I_idx, J_idx = Zygote.ignore() do
        I = Int[]
        J = Int[]
        for elem in elements
            nodes_elem = [elem[1], elem[2], elem[3]]
            dofs = Int[]
            for n in nodes_elem
                push!(dofs, 2*n-1)
                push!(dofs, 2*n)
            end
            for r in 1:6
                for c in 1:6
                    push!(I, dofs[r])
                    push!(J, dofs[c])
                end
            end
        end
        return I, J
    end
    
    V_vals = vcat([vec(k) for k in local_Ks]...)
    K = sparse(I_idx, J_idx, V_vals, n_dofs, n_dofs)
    
    # Load vector
    F = Zygote.ignore() do
        assemble_distributed_load_2d(nodes, elements, boundary_labels, q, L)
    end
    
    # Boundary conditions
    left_nodes = boundary_labels[:left]
    fixed_dofs = Int[]
    for node_idx in left_nodes
        push!(fixed_dofs, 2*node_idx - 1)
        push!(fixed_dofs, 2*node_idx)
    end
    
    penalty_base = 1e9
    K_pen_fixed = Zygote.ignore() do
        sparse(fixed_dofs, fixed_dofs, penalty_base, n_dofs, n_dofs)
    end
    
    sigma = L / (2 * nx)
    K_pen_support = apply_intermediate_support_penalty_2d(
        K, nodes, support_positions, L, penalty_base, sigma
    )
    
    # Solve
    K_total = Array(K + K_pen_fixed + K_pen_support)
    u = K_total \ F
    
    # Compliance
    compliance = dot(F, u)
    
    return compliance, u, nodes, elements
end

# ============================================================================
# 6. Post-processing Functions
# ============================================================================
function extract_tip_deflection_2d(u, nodes, L, h)
    """
    Extract tip deflection from 2D FEA solution.
    Returns vertical displacement at (L, h/2).
    """
    # Find node closest to (L, h/2)
    target_y = h / 2.0
    min_dist = Inf
    tip_node = 1
    
    for (idx, (x, y)) in enumerate(nodes)
        if abs(x - L) < 1e-6  # On right edge
            dist = abs(y - target_y)
            if dist < min_dist
                min_dist = dist
                tip_node = idx
            end
        end
    end
    
    # Vertical displacement DOF is 2*tip_node
    tip_deflection = u[2*tip_node]
    return tip_deflection
end

function compute_von_mises_stress_2d(u, nodes, elements, E, nu, b)
    """
    Compute von Mises stress field from displacement solution.
    Returns maximum von Mises stress.
    """
    # For each element, compute stress at centroid
    max_stress = 0.0
    
    for elem in elements
        i1, i2, i3 = elem[1], elem[2], elem[3]
        n1 = nodes[i1]
        n2 = nodes[i2]
        n3 = nodes[i3]
        
        # Element centroid
        x_c = (n1[1] + n2[1] + n3[1]) / 3.0
        y_c = (n1[2] + n2[2] + n3[2]) / 3.0
        
        # Get element displacement
        u_elem = [
            u[2*i1-1], u[2*i1],  # Node 1: ux, uy
            u[2*i2-1], u[2*i2],  # Node 2: ux, uy
            u[2*i3-1], u[2*i3]   # Node 3: ux, uy
        ]
        
        # Compute B matrix at centroid (simplified - CST has constant strain)
        x1, y1 = n1[1], n1[2]
        x2, y2 = n2[1], n2[2]
        x3, y3 = n3[1], n3[2]
        
        b1 = y2 - y3; b2 = y3 - y1; b3 = y1 - y2
        c1 = x3 - x2; c2 = x1 - x3; c3 = x2 - x1
        DoubleArea = (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
        Area = 0.5 * abs(DoubleArea) + 1e-10
        
        # B matrix
        B = [
            b1 0.0 b2 0.0 b3 0.0;
            0.0 c1 0.0 c2 0.0 c3;
            c1 b1 c2 b2 c3 b3
        ] .* (1.0 / DoubleArea)
        
        # Strain
        epsilon = B * u_elem
        eps_xx = epsilon[1]
        eps_yy = epsilon[2]
        gamma_xy = epsilon[3]
        
        # Stress (plane stress)
        factor = E / (1 - nu^2)
        sigma_xx = factor * (eps_xx + nu * eps_yy)
        sigma_yy = factor * (eps_yy + nu * eps_xx)
        tau_xy = factor * (1 - nu) / 2 * gamma_xy
        
        # Von Mises stress
        von_mises = sqrt(sigma_xx^2 + sigma_yy^2 - sigma_xx*sigma_yy + 3*tau_xy^2)
        max_stress = max(max_stress, abs(von_mises))
    end
    
    return max_stress
end

# ============================================================================
# 7. Gradient Wrapper
# ============================================================================
function ∇compliance_fea_2d(support_positions::Vector{Float64})
    """
    Compute gradient of compliance w.r.t. support positions using Zygote.
    """
    grads = Zygote.gradient(x -> solve_beam_compliance_2d(x), support_positions)[1]
    return grads
end

