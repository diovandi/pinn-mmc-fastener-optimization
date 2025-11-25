#!/usr/bin/env julia

using LinearAlgebra
using SparseArrays
using Zygote
using JSON
using Printf
using Dates

const JSON_PATH = joinpath(@__DIR__, "../../data/cad/lbracket.json")
const RESULT_DIR = joinpath(@__DIR__, "../../data/results")
const LOG_PATH = joinpath(RESULT_DIR, "lbracket_diff_fea_log.csv")

struct LBracketProblem
    coords::Vector{Float64}
    elements::Vector{Vector{Int}}
    fixed_dofs::Vector{Int}
    loads::Vector{Float64}
    centroids::Matrix{Float64}
    nu::Float64
    thick::Float64
    E_base::Float64
    E_bolt::Float64
    bolt_radius::Float64
end

function ensure_dirs()
    isdir(RESULT_DIR) || mkpath(RESULT_DIR)
end

function load_problem()::LBracketProblem
    data = JSON.parsefile(JSON_PATH)
    coords = Float64.(data["coords"])
    elements = [Int.(e) for e in data["elements"]]
    fixed_dofs = Int.(data["fixed_dofs"])
    load_dof = Int(data["load_dof"])
    nu = Float64(data["nu"])
    thick = Float64(data["thick"])
    E = Float64(data["E"])
    E_bolt = 200000.0
    bolt_radius = 15.0

    loads = zeros(length(coords))
    loads[load_dof] = -1000.0

    centroids = get_centroids(coords, elements)
    return LBracketProblem(coords, elements, fixed_dofs, loads, centroids, nu, thick, E, E_bolt, bolt_radius)
end

function get_centroids(flat_coords, elements)
    centroids = zeros(length(elements), 2)
    for (i, elem) in enumerate(elements)
        p1 = flat_coords[(2*elem[1]-1):(2*elem[1])]
        p2 = flat_coords[(2*elem[2]-1):(2*elem[2])]
        p3 = flat_coords[(2*elem[3]-1):(2*elem[3])]
        centroids[i, :] = (p1 + p2 + p3) / 3.0
    end
    return centroids
end

function enforce_lbracket_bounds(coords::Vector{Float64})
    THICK = 25.0
    MAX_DIM = 100.0
    PADDING = 5.0

    n_screws = Int(length(coords) ÷ 2)
    projected = map(1:n_screws) do i
        x = clamp(coords[2*i-1], PADDING, MAX_DIM - PADDING)
        y = clamp(coords[2*i], PADDING, MAX_DIM - PADDING)

        if x > THICK && y > THICK
            if (x - THICK) < (y - THICK)
                x = THICK - PADDING
            else
                y = THICK - PADDING
            end
        end

        return [x, y]
    end

    return vcat(projected...)
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
    n_screws = Int(length(screw_coords) ÷ 2)
    gain = E_bolt - E_base
    denom = 2 * radius^2 + 1e-6
    cx = centroids[:, 1]
    cy = centroids[:, 2]
    contributions = zeros(n_elems)

    for i in 1:n_screws
        sx = screw_coords[2*i-1]
        sy = screw_coords[2*i]
        dist_sq = (cx .- sx).^2 .+ (cy .- sy).^2
        contributions = contributions .+ gain .* exp.(-dist_sq ./ denom)
    end

    return E_base .+ contributions
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
            dofs = [2*elem[1]-1, 2*elem[1], 2*elem[2]-1, 2*elem[2],
                    2*elem[3]-1, 2*elem[3]]
            for r in 1:6, c in 1:6
                push!(I, dofs[r])
                push!(J, dofs[c])
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

function compliance_from_screws(screws, prob::LBracketProblem)
    safe = enforce_lbracket_bounds(screws)
    Ef = project_material_properties(safe, prob.centroids, prob.E_base, prob.E_bolt, prob.bolt_radius)
    c = solve_compliance(prob.coords, prob.elements, Ef, prob.nu, prob.thick, prob.loads, prob.fixed_dofs)
    return c, safe
end

function run_diff_fea_log(; iters=25, lr=1.0, log_path=joinpath(RESULT_DIR, "lbracket_diff_fea_log.csv"))
    ensure_dirs()
    prob = load_problem()
    screws = Float64[20.0, 20.0, 15.0, 80.0]
    m = zeros(length(screws))
    v = zeros(length(screws))
    β1, β2 = 0.9, 0.999
    ϵ = 1e-8

    open(log_path, "w") do io
        println(io, "method,iter,compliance,s1_x,s1_y,s2_x,s2_y,wall_time_ms,timestamp")
        for iter in 0:iters
            iter_start = time()
            c, safe = compliance_from_screws(screws, prob)
            elapsed_ms = (time() - iter_start) * 1000.0
            @printf(io, "diff_fea,%d,%.6f,%.4f,%.4f,%.4f,%.4f,%.3f,%s\n",
                    iter, c, safe[1], safe[2], safe[3], safe[4], elapsed_ms,
                    Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"))

            iter == iters && break

            loss_fn = s -> begin
                val, _ = compliance_from_screws(s, prob)
                return val
            end
            grads = Zygote.gradient(loss_fn, screws)[1]
            grads = clamp.(grads, -100.0, 100.0)

            m = β1 .* m .+ (1 - β1) .* grads
            v = β2 .* v .+ (1 - β2) .* (grads .^ 2)
            m_hat = m ./ (1 - β1^(iter + 1))
            v_hat = v ./ (1 - β2^(iter + 1))

            screws .-= lr .* m_hat ./ (sqrt.(v_hat) .+ ϵ)
            screws = enforce_lbracket_bounds(screws)
        end
    end

    println("✅ Saved diff-FEA benchmark log to $log_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_diff_fea_log()
end

