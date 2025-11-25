#!/usr/bin/env julia

using Random
using CSV
using DataFrames
using Printf
using Dates

const ROOT_DIR = normpath(joinpath(@__DIR__, "../../.."))
const APPROACH_A_DIR = normpath(joinpath(@__DIR__, "../../../approach_a_pinn"))
const OUTPUT_ROOT = normpath(joinpath(ROOT_DIR, "../data/results/multi_geom_training"))
mkpath(OUTPUT_ROOT)

include(joinpath(APPROACH_A_DIR, "LBracketBenchmarkLogger.jl"))

struct RolloutArgs
    load_case::Symbol
    samples::Int
    seed::Int
    output_path::String
end

function parse_args()::RolloutArgs
    load_case = :vertical_tip
    samples = 100
    seed = 42
    output_path = joinpath(OUTPUT_ROOT, "l_bracket_rollout.csv")

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg in ("--load-case", "-l")
            i += 1
            load_case = Symbol(ARGS[i])
        elseif arg in ("--samples", "-n")
            i += 1
            samples = parse(Int, ARGS[i])
        elseif arg in ("--seed", "-s")
            i += 1
            seed = parse(Int, ARGS[i])
        elseif arg in ("--output", "-o")
            i += 1
            output_path = ARGS[i]
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end

    return RolloutArgs(load_case, samples, seed, output_path)
end

function with_load_case(prob::LBracketProblem, case::Symbol)
    load_idx = findfirst(!iszero, prob.loads)
    load_mag = prob.loads[load_idx]
    loads = zeros(length(prob.loads))

    if case == :vertical_tip
        loads[load_idx] = load_mag
    elseif case == :horizontal_tip
        horizontal_idx = max(load_idx - 1, 1)
        loads[horizontal_idx] = -abs(load_mag)
    else
        error("Unsupported load case: $case")
    end

    return LBracketProblem(
        prob.coords,
        prob.elements,
        prob.fixed_dofs,
        loads,
        prob.centroids,
        prob.nu,
        prob.thick,
        prob.E_base,
        prob.E_bolt,
        prob.bolt_radius,
    )
end

function random_layout(rng::AbstractRNG)
    screws = zeros(4)
    for s in 1:2
        if rand(rng) > 0.5
            screws[2s - 1] = rand(rng) * 25.0
            screws[2s] = rand(rng) * 100.0
        else
            screws[2s - 1] = rand(rng) * 100.0
            screws[2s] = rand(rng) * 25.0
        end
    end
    return enforce_lbracket_bounds(screws)
end

function rollout(args::RolloutArgs)
    rng = MersenneTwister(args.seed)
    base_prob = load_problem()
    prob = with_load_case(base_prob, args.load_case)

    rows = Vector{NamedTuple}(undef, args.samples)
    for i in 1:args.samples
        screws = random_layout(rng)
        compliance, safe = compliance_from_screws(screws, prob)
        rows[i] = (
            geometry = "l_bracket",
            load_case = String(args.load_case),
            sample_id = i,
            s1_x = safe[1],
            s1_y = safe[2],
            s2_x = safe[3],
            s2_y = safe[4],
            compliance = compliance,
            timestamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        )
        @printf("Generated L-bracket sample %d/%d -> C = %.3f\n", i, args.samples, compliance)
    end

    df = DataFrame(rows)
    CSV.write(args.output_path, df)
    println("✅ Saved rollout to $(args.output_path)")
end

function main()
    try
        args = parse_args()
        rollout(args)
    catch err
        @error "Rollout failed" exception=err
        rethrow(err)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

