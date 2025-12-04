using LinearAlgebra
using Zygote
using SparseArrays
using Printf

# ============================================================================
# Cantilever Beam Differentiable FEA - 1D Euler-Bernoulli Beam Elements
# ============================================================================

# --- Material and Geometry Constants ---
const E_1D = 210e9  # Young's modulus (Pa) - Steel
const nu_1D = 0.3   # Poisson's ratio
const b_1D = 0.1    # Beam width (m)
const h_1D = 0.05   # Beam height (m)
const L_1D = 1.0    # Beam length (m)
const q_1D = 1000.0 # Distributed load (N/m)

# Second moment of area for rectangular cross-section
const I_1D = b_1D * h_1D^3 / 12.0

# ============================================================================
# 1. Beam Element Stiffness Matrix (Euler-Bernoulli)
# ============================================================================
# 4 DOF per element: [u1, θ1, u2, θ2]
# where u = vertical displacement, θ = rotation
function element_stiffness_beam(L_elem, E, I)
    """
    Compute 4x4 Euler-Bernoulli beam element stiffness matrix.
    L_elem: element length
    """
    # Element stiffness matrix for Euler-Bernoulli beam
    # K = (E*I/L^3) * [12    6L   -12    6L  ]
    #                  [6L   4L^2  -6L   2L^2 ]
    #                  [-12  -6L    12   -6L  ]
    #                  [6L   2L^2  -6L   4L^2 ]
    
    L2 = L_elem^2
    L3 = L_elem^3
    factor = E * I / L3
    
    K = factor * [
        12.0    6*L_elem  -12.0    6*L_elem;
        6*L_elem  4*L2    -6*L_elem  2*L2;
        -12.0   -6*L_elem   12.0   -6*L_elem;
        6*L_elem   2*L2    -6*L_elem  4*L2
    ]
    
    return K
end

# ============================================================================
# 2. Consistent Load Vector for Distributed Load
# ============================================================================
function element_load_vector_distributed(L_elem, q)
    """
    Consistent load vector for uniformly distributed load q.
    Returns 4x1 vector: [F1, M1, F2, M2]
    """
    # For uniformly distributed load q on a beam element:
    # F1 = q*L/2, M1 = q*L^2/12
    # F2 = q*L/2, M2 = -q*L^2/12
    
    F = [
        q * L_elem / 2.0;
        q * L_elem^2 / 12.0;
        q * L_elem / 2.0;
        -q * L_elem^2 / 12.0
    ]
    
    return F
end

# ============================================================================
# 3. Mesh Generation
# ============================================================================
function create_beam_mesh(L, n_elem)
    """
    Create 1D beam mesh with n_elem elements.
    Returns: nodes (positions), elements (connectivity)
    """
    nodes = collect(range(0.0, L, length=n_elem+1))
    elements = [(i, i+1) for i in 1:n_elem]
    
    return nodes, elements
end

# ============================================================================
# 4. Stiffness Matrix Assembly
# ============================================================================
function assemble_stiffness(nodes, elements, E, I)
    """
    Assemble global stiffness matrix from element matrices.
    """
    n_nodes = length(nodes)
    n_dofs = 2 * n_nodes  # 2 DOF per node: u and θ
    n_elem = length(elements)
    
    # Compute element stiffness matrices (immutable)
    local_Ks = map(elements) do elem
        i1, i2 = elem[1], elem[2]
        x1, x2 = nodes[i1], nodes[i2]
        L_elem = abs(x2 - x1)
        element_stiffness_beam(L_elem, E, I)
    end
    
    # Assembly indices (non-differentiable)
    I_idx, J_idx = Zygote.ignore() do
        I = Int[]
        J = Int[]
        
        for (idx, elem) in enumerate(elements)
            i1, i2 = elem[1], elem[2]
            # DOF mapping: node i -> [2*i-1 (u), 2*i (θ)]
            dofs = [2*i1-1, 2*i1, 2*i2-1, 2*i2]
            
            # 4x4 block
            for r in 1:4
                for c in 1:4
                    push!(I, dofs[r])
                    push!(J, dofs[c])
                end
            end
        end
        
        return I, J
    end
    
    # Flatten values (differentiable)
    V_vals = vcat([vec(k) for k in local_Ks]...)
    
    # Global stiffness matrix
    K = sparse(I_idx, J_idx, V_vals, n_dofs, n_dofs)
    
    return K
end

# ============================================================================
# 5. Load Vector Assembly
# ============================================================================
function assemble_force_loads(nodes, elements, q)
    """
    Assemble global load vector from distributed load q.
    """
    n_nodes = length(nodes)
    n_dofs = 2 * n_nodes
    F = zeros(n_dofs)
    
    for elem in elements
        i1, i2 = elem[1], elem[2]
        x1, x2 = nodes[i1], nodes[i2]
        L_elem = abs(x2 - x1)
        
        # Element load vector
        F_elem = element_load_vector_distributed(L_elem, q)
        
        # Map to global DOFs
        dofs = [2*i1-1, 2*i1, 2*i2-1, 2*i2]
        F[dofs] .+= F_elem
    end
    
    return F
end

# ============================================================================
# 6. Differentiable Compliance Solver
# ============================================================================
function solve_beam_compliance_1d(support_positions::Vector{Float64}, 
                                  L=L_1D, n_elem=25, E=E_1D, I=I_1D, q=q_1D)
    """
    Solve 1D beam FEA and compute compliance.
    support_positions: vector of support positions along beam (in [0.1L, 0.9L])
    Returns: compliance C = F' * u
    """
    # Create mesh (constant, not dependent on support positions)
    nodes, elements = Zygote.ignore() do
        create_beam_mesh(L, n_elem)
    end
    n_nodes = length(nodes)
    n_dofs = 2 * n_nodes
    
    # Assemble stiffness and load
    K = assemble_stiffness(nodes, elements, E, I)
    # Load vector is constant (not dependent on support positions), so we can ignore it for AD
    F = Zygote.ignore() do
        assemble_force_loads(nodes, elements, q)
    end
    
    # Boundary conditions
    # Fixed support at x=0 (left end): u=0, θ=0
    base_fixed_dofs = [1, 2]  # DOF 1 and 2 for node 1
    
    # Intermediate supports: use continuous penalty method
    # Apply penalty that varies smoothly with distance from support position
    # This makes the constraint differentiable w.r.t. support position
    penalty_base = 1e9
    sigma = L / (2 * n_elem)  # Characteristic length scale
    
    # Build penalty matrix contributions (immutable)
    # Fixed support penalty
    I_pen_fixed = base_fixed_dofs
    J_pen_fixed = base_fixed_dofs
    V_pen_fixed = [penalty_base for _ in base_fixed_dofs]
    
    # Intermediate support penalties (differentiable w.r.t. position)
    # Use a simpler approach: apply penalty to all valid nodes with weights
    # This avoids complex list comprehensions that Zygote struggles with
    valid_node_indices, valid_u_dofs = Zygote.ignore() do
        valid_nodes = [i for i in 1:n_nodes if nodes[i] >= 0.1*L && nodes[i] <= 0.9*L]
        valid_dofs = [2*i - 1 for i in valid_nodes if (2*i - 1) ∉ base_fixed_dofs]
        (valid_nodes, valid_dofs)
    end
    
    # Compute weights for each support position and valid DOF (differentiable)
    # This is the only part that needs to be differentiable
    # Build weights matrix immutably
    weights_matrix = [
        begin
            node_idx = (dof + 1) ÷ 2
            node_pos = nodes[node_idx]
            pos_clamped = clamp(pos, 0.1*L, 0.9*L)
            dist_sq = (node_pos - pos_clamped)^2
            exp(-dist_sq / (2 * sigma^2))
        end
        for dof in valid_u_dofs, pos in support_positions
    ]
    
    # Sum weights across all support positions for each DOF
    total_weights = sum(weights_matrix, dims=2)[:]  # Sum over support positions
    
    # Build penalty matrix (indices fixed, values differentiable)
    I_pen_support = valid_u_dofs
    J_pen_support = valid_u_dofs
    V_pen_support = penalty_base .* total_weights
    
    # Combine with fixed penalties
    I_pen = [I_pen_fixed; I_pen_support]
    J_pen = [J_pen_fixed; J_pen_support]
    V_pen = [V_pen_fixed; V_pen_support]
    
    # Build sparse penalty matrix (indices are non-differentiable, values are)
    I_idx, J_idx = Zygote.ignore() do
        (I_pen, J_pen)
    end
    K_pen = sparse(I_idx, J_idx, V_pen, n_dofs, n_dofs)
    
    # Solve
    K_total = Array(K + K_pen)  # Dense for Zygote stability
    u = K_total \ F
    
    # Compliance
    compliance = dot(F, u)
    
    return compliance
end

# ============================================================================
# Extended Function with Tip Deflection
# ============================================================================
function solve_beam_compliance_extended_1d(support_positions::Vector{Float64}, 
                                           L=L_1D, n_elem=25, E=E_1D, I=I_1D, q=q_1D)
    """
    Solve 1D beam FEA and return compliance, displacement vector, and nodes.
    Used for extracting tip deflection and other post-processing.
    """
    # Create mesh (constant, not dependent on support positions)
    nodes, elements = Zygote.ignore() do
        create_beam_mesh(L, n_elem)
    end
    n_nodes = length(nodes)
    n_dofs = 2 * n_nodes
    
    # Assemble stiffness and load
    K = assemble_stiffness(nodes, elements, E, I)
    F = Zygote.ignore() do
        assemble_force_loads(nodes, elements, q)
    end
    
    # Boundary conditions (same as solve_beam_compliance_1d)
    base_fixed_dofs = [1, 2]
    penalty_base = 1e9
    sigma = L / (2 * n_elem)
    
    I_pen_fixed = base_fixed_dofs
    J_pen_fixed = base_fixed_dofs
    V_pen_fixed = [penalty_base for _ in base_fixed_dofs]
    
    valid_node_indices, valid_u_dofs = Zygote.ignore() do
        valid_nodes = [i for i in 1:n_nodes if nodes[i] >= 0.1*L && nodes[i] <= 0.9*L]
        valid_dofs = [2*i - 1 for i in valid_nodes if (2*i - 1) ∉ base_fixed_dofs]
        (valid_nodes, valid_dofs)
    end
    
    weights_matrix = [
        begin
            node_idx = (dof + 1) ÷ 2
            node_pos = nodes[node_idx]
            pos_clamped = clamp(pos, 0.1*L, 0.9*L)
            dist_sq = (node_pos - pos_clamped)^2
            exp(-dist_sq / (2 * sigma^2))
        end
        for dof in valid_u_dofs, pos in support_positions
    ]
    
    total_weights = sum(weights_matrix, dims=2)[:]
    I_pen_support = valid_u_dofs
    J_pen_support = valid_u_dofs
    V_pen_support = penalty_base .* total_weights
    
    I_pen = [I_pen_fixed; I_pen_support]
    J_pen = [J_pen_fixed; J_pen_support]
    V_pen = [V_pen_fixed; V_pen_support]
    
    I_idx, J_idx = Zygote.ignore() do
        (I_pen, J_pen)
    end
    K_pen = sparse(I_idx, J_idx, V_pen, n_dofs, n_dofs)
    
    # Solve
    K_total = Array(K + K_pen)
    u = K_total \ F
    
    # Compliance
    compliance = dot(F, u)
    
    return compliance, u, nodes
end

# ============================================================================
# Extract Tip Deflection
# ============================================================================
function extract_tip_deflection_1d(u, nodes)
    """
    Extract tip deflection from 1D FEA solution.
    u: displacement vector (2 DOF per node: u, θ)
    nodes: node positions
    Returns: tip deflection (vertical displacement at x = L)
    """
    n_nodes = length(nodes)
    # Tip node is the last node (x = L)
    tip_node = n_nodes
    # Vertical displacement DOF is 2*tip_node - 1
    tip_deflection = u[2*tip_node - 1]
    return tip_deflection
end

# ============================================================================
# Gradient Wrapper
# ============================================================================
function ∇compliance_fea_1d(support_positions::Vector{Float64})
    """
    Compute gradient of compliance w.r.t. support positions using Zygote.
    """
    grads = Zygote.gradient(x -> solve_beam_compliance_1d(x), support_positions)[1]
    return grads
end

# Backward compatibility aliases
const solve_beam_compliance = solve_beam_compliance_1d
const solve_beam_compliance_extended = solve_beam_compliance_extended_1d
const extract_tip_deflection = extract_tip_deflection_1d
const ∇compliance_fea = ∇compliance_fea_1d

