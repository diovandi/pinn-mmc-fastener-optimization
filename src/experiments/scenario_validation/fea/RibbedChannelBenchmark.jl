#!/usr/bin/env julia

using LinearAlgebra
using SparseArrays

const ROOT_DIR = normpath(joinpath(@__DIR__, "../../.."))
const LENGTH = 160.0
const HEIGHT = 50.0
const SLOT_X = (60.0, 100.0)
const SLOT_Y = (15.0, 35.0)
const NU = 0.33
const E_BASE = 70e3
const E_BOLT = 200e3
const BOLT_RADIUS = 10.0
const THICK_BASE = 4.0

const DEFAULT_NX = 64
const DEFAULT_NY = 20

struct ChannelMesh
    coords::Vector{Float64}
    elements::Vector{NTuple{3,Int}}
    element_centroids::Matrix{Float64}
    element_thickness::Vector{Float64}
    fixed_dofs::Vector{Int}
    load_upward::Vector{Float64}
    load_shear::Vector{Float64}
end

function build_channel_mesh(; nx::Int=DEFAULT_NX, ny::Int=DEFAULT_NY)::ChannelMesh
    dx = LENGTH / nx
    dy = HEIGHT / ny
    nodes = Vector{Tuple{Float64,Float64}}()
    for j in 0:ny
        for i in 0:nx
            push!(nodes, (i * dx, j * dy))
        end
    end

    elements = NTuple{3,Int}[]
    to_index = (i,j) -> j*(nx+1) + i + 1
    for j in 0:(ny-1), i in 0:(nx-1)
        n1 = to_index(i, j)
        n2 = to_index(i+1, j)
        n3 = to_index(i+1, j+1)
        n4 = to_index(i, j+1)
        push!(elements, (n1, n2, n3))
        push!(elements, (n1, n3, n4))
    end

    flat_coords = vcat([Float64[x, y] for (x,y) in nodes]...)
    centroids = zeros(length(elements), 2)
    thickness = fill(THICK_BASE, length(elements))
    for (idx, elem) in enumerate(elements)
        p1 = flat_coords[(2*elem[1]-1):(2*elem[1])]
        p2 = flat_coords[(2*elem[2]-1):(2*elem[2])]
        p3 = flat_coords[(2*elem[3]-1):(2*elem[3])]
        centroid = (p1 + p2 + p3) ./ 3
        centroids[idx, :] = centroid
        if SLOT_X[1] ≤ centroid[1] ≤ SLOT_X[2] && SLOT_Y[1] ≤ centroid[2] ≤ SLOT_Y[2]
            thickness[idx] = THICK_BASE * 0.05
        end
    end

    fixed_nodes = Int[]
    for (idx, (x, y)) in enumerate(nodes)
        if x ≤ dx / 4
            push!(fixed_nodes, idx)
        elseif SLOT_X[1] < x < SLOT_X[2] && SLOT_Y[1] < y < SLOT_Y[2]
            push!(fixed_nodes, idx)
        end
    end
    fixed_dofs = reduce(vcat, ([2*n - 1, 2*n] for n in fixed_nodes); init=Int[])

    ndof = 2 * length(nodes)
    load_upward = zeros(ndof)
    tip_nodes = [idx for (idx,(x,_)) in enumerate(nodes) if x ≥ LENGTH - dx/2]
    for n in tip_nodes
        load_upward[2n] -= 600.0 / max(length(tip_nodes), 1)
    end

    load_shear = zeros(ndof)
    top_nodes = [idx for (idx,(_,y)) in enumerate(nodes) if y ≥ HEIGHT - dy/2]
    for n in top_nodes
        load_shear[2n - 1] += 400.0 / max(length(top_nodes), 1)
    end

    return ChannelMesh(flat_coords, elements, centroids, thickness, fixed_dofs, load_upward, load_shear)
end

function element_routine_cst(n1, n2, n3, E, nu, thickness)
    b1 = n2[2] - n3[2]; b2 = n3[2] - n1[2]; b3 = n1[2] - n2[2]
    c1 = n3[1] - n2[1]; c2 = n1[1] - n3[1]; c3 = n2[1] - n1[1]
    double_area = (n1[1]*(n2[2]-n3[2]) + n2[1]*(n3[2]-n1[2]) + n3[1]*(n1[2]-n2[2]))
    area = 0.5 * max(abs(double_area), 1e-6)
    factor = E / ((1 - nu^2) + 1e-6)
    B = [b1 0 b2 0 b3 0;
         0 c1 0 c2 0 c3;
         c1 b1 c2 b2 c3 b3] .* (1.0 / (2 * area))
    D = [1 nu 0; nu 1 0; 0 0 (1 - nu) / 2] .* factor
    return (transpose(B) * D * B) .* (area * thickness)
end

function assemble_global_K(mesh::ChannelMesh, Ef::Vector{Float64})
    n_dofs = length(mesh.coords)
    I = Int[]
    J = Int[]
    V = Float64[]
    for (idx, elem) in enumerate(mesh.elements)
        n1 = mesh.coords[(2*elem[1]-1):(2*elem[1])]
        n2 = mesh.coords[(2*elem[2]-1):(2*elem[2])]
        n3 = mesh.coords[(2*elem[3]-1):(2*elem[3])]
        k_elem = element_routine_cst(n1, n2, n3, max(Ef[idx], 100.0), NU, mesh.element_thickness[idx])
        dofs = [2*elem[1]-1, 2*elem[1], 2*elem[2]-1, 2*elem[2], 2*elem[3]-1, 2*elem[3]]
        for r in 1:6, c in 1:6
            push!(I, dofs[r])
            push!(J, dofs[c])
            push!(V, k_elem[r, c])
        end
    end
    return sparse(I, J, V, n_dofs, n_dofs)
end

function project_material_field(screws::Vector{Float64}, centroids::Matrix{Float64})
    n_screws = Int(length(screws) ÷ 2)
    Ef = fill(E_BASE, size(centroids, 1))
    cx = centroids[:, 1]
    cy = centroids[:, 2]
    gain = E_BOLT - E_BASE
    denom = 2 * BOLT_RADIUS^2 + 1e-6
    for i in 1:n_screws
        sx = screws[2i - 1]
        sy = screws[2i]
        dist_sq = (cx .- sx).^2 .+ (cy .- sy).^2
        Ef .+= gain .* exp.(-dist_sq ./ denom)
    end
    return Ef
end

function clamp_screws(layout::Vector{Float64})
    n = Int(length(layout) ÷ 2)
    clamped = similar(layout)
    for i in 1:n
        clamped[2i - 1] = clamp(layout[2i - 1], 15.0, LENGTH - 15.0)
        clamped[2i] = clamp(layout[2i], 5.0, HEIGHT - 5.0)
    end
    return clamped
end

function solve_compliance(mesh::ChannelMesh, screws::Vector{Float64}, case::Symbol)
    Ef = project_material_field(screws, mesh.element_centroids)
    K = assemble_global_K(mesh, Ef)
    penalty = sparse(mesh.fixed_dofs, mesh.fixed_dofs, 1e9, length(mesh.coords), length(mesh.coords))
    K_dense = Array(K + penalty) + Diagonal(fill(1e-6, length(mesh.coords)))
    loads = if case == :upward_tip
        mesh.load_upward
    elseif case == :lateral_shear
        mesh.load_shear
    else
        error("Unsupported load case: $case")
    end
    u = K_dense \ loads
    return dot(loads, u)
end

if abspath(PROGRAM_FILE) == @__FILE__
    mesh = build_channel_mesh()
    screws = clamp_screws([35.0, 12.0, 80.0, 25.0, 125.0, 30.0])
    println("Sample compliance (upward_tip) = ", solve_compliance(mesh, screws, :upward_tip))
    println("Sample compliance (lateral_shear) = ", solve_compliance(mesh, screws, :lateral_shear))
end

