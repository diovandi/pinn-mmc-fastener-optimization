#!/usr/bin/env julia
"""
Differentiable (or at least auto-diff friendly) FEA benchmark for the tapered
cantilever plate scenario. This script mirrors the structure of
`approach_a_pinn/LBracketBenchmarkLogger.jl` but generates its own rectangular
mesh so no existing files need to change.
"""

using LinearAlgebra
using SparseArrays
using Printf
using Dates
using CSV
using DataFrames

const SCENARIO_ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(SCENARIO_ROOT, "results", "fea_new_part")
mkpath(RESULTS_DIR)

const PLATE_LENGTH = 140.0
const PLATE_HEIGHT = 60.0
const CLAMP_REGION = 10.0
const NU = 0.33
const E_BASE = 70e3
const E_BOLT = 200e3
const BOLT_RADIUS = 12.0
const TIP_FORCE = 450.0
const TORSION = 25.0
const DEFAULT_NX = 40
const DEFAULT_NY = 16

struct PlateMesh
    coords::Vector{Float64}
    elements::Vector{NTuple{3,Int}}
    element_centroids::Matrix{Float64}
    element_thickness::Vector{Float64}
    fixed_dofs::Vector{Int}
    loads::Vector{Float64}
end

function build_plate_mesh(; nx::Int=DEFAULT_NX, ny::Int=DEFAULT_NY)::PlateMesh
    dx = PLATE_LENGTH / nx
    dy = PLATE_HEIGHT / ny
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
    thickness = zeros(length(elements))
    for (idx, elem) in enumerate(elements)
        p1 = flat_coords[(2*elem[1]-1):(2*elem[1])]
        p2 = flat_coords[(2*elem[2]-1):(2*elem[2])]
        p3 = flat_coords[(2*elem[3]-1):(2*elem[3])]
        centroid = (p1 + p2 + p3) ./ 3
        centroids[idx, :] = centroid
        taper_ratio = centroid[1] / max(PLATE_LENGTH, 1e-6)
        thickness[idx] = 6.0 - 3.0 * taper_ratio
    end

    fixed_nodes = findall(p -> p[1] ≤ CLAMP_REGION + 1e-6, nodes)
    fixed_dofs = reduce(vcat, ([2*n - 1, 2*n] for n in fixed_nodes); init=Int[])

    loads = zeros(Float64, length(flat_coords))
    tip_nodes = findall(p -> p[1] ≥ (PLATE_LENGTH - dx / 2), nodes)
    distributed_force = TIP_FORCE / max(length(tip_nodes), 1)
    for n in tip_nodes
        loads[2n] -= distributed_force
    end
    top_nodes = findall(p -> p[1] ≥ PLATE_LENGTH - dx && p[2] ≥ PLATE_HEIGHT / 2, nodes)
    bottom_nodes = findall(p -> p[1] ≥ PLATE_LENGTH - dx && p[2] < PLATE_HEIGHT / 2, nodes)
    torque_arm = PLATE_HEIGHT / 2
    torsion_force = TORSION / max(torque_arm * length(top_nodes), 1)
    for n in top_nodes
        loads[2n] -= torsion_force
    end
    for n in bottom_nodes
        loads[2n] += torsion_force
    end

    return PlateMesh(flat_coords, elements, centroids, thickness, fixed_dofs, loads)
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

function project_material_field(
    screws::Vector{Float64},
    centroids::Matrix{Float64},
    thickness::Vector{Float64}
)::Tuple{Vector{Float64},Vector{Float64}}
    n_screws = Int(length(screws) ÷ 2)
    cx = centroids[:, 1]
    cy = centroids[:, 2]
    Ef = fill(E_BASE, length(cx))
    for i in 1:n_screws
        sx = screws[2i - 1]
        sy = screws[2i]
        dist_sq = (cx .- sx).^2 .+ (cy .- sy).^2
        Ef .+= (E_BOLT - E_BASE) .* exp.(-dist_sq ./ (2 * BOLT_RADIUS^2 + 1e-6))
    end
    return Ef, thickness
end

function assemble_global_K(mesh::PlateMesh, Ef::Vector{Float64})
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

function solve_compliance(mesh::PlateMesh, screws::Vector{Float64})
    Ef, thickness = project_material_field(screws, mesh.element_centroids, mesh.element_thickness)
    K = assemble_global_K(mesh, Ef)
    penalty = sparse(mesh.fixed_dofs, mesh.fixed_dofs, 1e9, length(mesh.coords), length(mesh.coords))
    K_dense = Array(K + penalty) + Diagonal(fill(1e-6, length(mesh.coords)))
    u = K_dense \ mesh.loads
    return dot(mesh.loads, u)
end

function clamp_screws(screws::Vector{Float64})
    n = Int(length(screws) ÷ 2)
    clamped = similar(screws)
    for i in 1:n
        clamped[2i - 1] = clamp(screws[2i - 1], CLAMP_REGION + 5.0, PLATE_LENGTH - 5.0)
        clamped[2i] = clamp(screws[2i], 5.0, PLATE_HEIGHT - 5.0)
    end
    return clamped
end

function evaluate_csv(input_csv::AbstractString; output_path::AbstractString="")
    mesh = build_plate_mesh()
    rows = DataFrame(CSV.File(input_csv))
    output_path = isempty(output_path) ? joinpath(RESULTS_DIR, "tapered_plate_diff_fea_log.csv") : output_path
    open(output_path, "w") do io
        println(io, "source,iter,compliance,s1_x,s1_y,s2_x,s2_y,wall_time_ms,timestamp")
        for (idx, row) in enumerate(eachrow(rows))
            screws = clamp_screws(Float64[row.s1_x, row.s1_y, row.s2_x, row.s2_y])
            start_t = time()
            c = solve_compliance(mesh, screws)
            elapsed = (time() - start_t) * 1000
            @printf(io, "pinn_layout,%d,%.6f,%.4f,%.4f,%.4f,%.4f,%.3f,%s\n",
                idx - 1, c, screws[1], screws[2], screws[3], screws[4], elapsed,
                Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"))
        end
    end
    println("✅ Saved FEA benchmark to $output_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    input_csv = get(ENV, "PINN_LAYOUTS", joinpath(SCENARIO_ROOT, "results", "pinn_new_part", "pinn_tapered_plate_predictions.csv"))
    evaluate_csv(input_csv)
end

