#!/usr/bin/env julia

using Random
using CSV
using DataFrames
using Printf
using Dates

const ROOT_DIR = normpath(joinpath(@__DIR__, "../../.."))
const OUTPUT_ROOT = normpath(joinpath(ROOT_DIR, "../data/results/multi_geom_training"))
const FEA_DIR = normpath(joinpath(@__DIR__, "../fea"))
mkpath(OUTPUT_ROOT)

include(joinpath(FEA_DIR, "RibbedChannelBenchmark.jl"))

struct RolloutArgs
    load_case::Symbol
    samples::Int
    seed::Int
    output_path::String
end

function parse_args()::RolloutArgs
    load_case = :upward_tip
    samples = 200
    seed = 2025
    output_path = joinpath(OUTPUT_ROOT, "ribbed_channel_rollout.csv")

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

const ROWS = [
    35.0,
    80.0,
    125.0,
]
const ROW_Y_RANGE = (8.0, 42.0)
const ROW_SPACING = 12.0

function enumerate_positions()
    y_vals = ROW_Y_RANGE[1]:ROW_SPACING:ROW_Y_RANGE[2]
    return [[(x, y) for y in y_vals] for x in ROWS]
end

function sample_layout(rng::AbstractRNG, positions)
    coords = Float64[]
    for row in positions
        pos = row[rand(rng, 1:length(row))]
        push!(coords, pos[1])
        push!(coords, pos[2])
    end
    return clamp_screws(coords)
end

function rollout(args::RolloutArgs)
    rng = MersenneTwister(args.seed)
    mesh = build_channel_mesh()
    positions = enumerate_positions()

    rows = Vector{NamedTuple}(undef, args.samples)
    for i in 1:args.samples
        layout = sample_layout(rng, positions)
        compliance = solve_compliance(mesh, layout, args.load_case)
        rows[i] = (
            geometry = "ribbed_channel",
            load_case = String(args.load_case),
            sample_id = i,
            s1_x = layout[1],
            s1_y = layout[2],
            s2_x = layout[3],
            s2_y = layout[4],
            s3_x = layout[5],
            s3_y = layout[6],
            compliance = compliance,
            timestamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        )
        @printf("Generated ribbed channel sample %d/%d -> C = %.3f\n", i, args.samples, compliance)
    end

    df = DataFrame(rows)
    CSV.write(args.output_path, df)
    println("✅ Saved rollout to $(args.output_path)")
end

function main()
    args = parse_args()
    rollout(args)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

