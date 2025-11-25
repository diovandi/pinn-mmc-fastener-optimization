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

include(joinpath(FEA_DIR, "NewPartBenchmark.jl"))

struct RolloutArgs
    samples::Int
    seed::Int
    output_path::String
end

function parse_args()::RolloutArgs
    samples = 150
    seed = 123
    output_path = joinpath(OUTPUT_ROOT, "tapered_plate_rollout.csv")

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg in ("--samples", "-n")
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

    return RolloutArgs(samples, seed, output_path)
end

const ROW_A_X = 20.0
const ROW_B_X = 80.0
const ROW_Y_MIN = 12.0
const ROW_Y_MAX = 48.0
const ROW_SPACING = 15.0

function enumerate_rows()
    y_vals = ROW_Y_MIN:ROW_SPACING:ROW_Y_MAX
    row_a = [(ROW_A_X, y) for y in y_vals]
    row_b = [(ROW_B_X, y) for y in y_vals]
    return row_a, row_b
end

function rollout(args::RolloutArgs)
    rng = MersenneTwister(args.seed)
    row_a, row_b = enumerate_rows()
    mesh = build_plate_mesh()

    rows = Vector{NamedTuple}(undef, args.samples)
    for i in 1:args.samples
        s1 = row_a[rand(rng, 1:length(row_a))]
        s2 = row_b[rand(rng, 1:length(row_b))]
        layout = clamp_screws([s1[1], s1[2], s2[1], s2[2]])
        compliance = solve_compliance(mesh, layout)
        rows[i] = (
            geometry = "tapered_plate",
            load_case = "combined_tip",
            sample_id = i,
            s1_x = layout[1],
            s1_y = layout[2],
            s2_x = layout[3],
            s2_y = layout[4],
            compliance = compliance,
            timestamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        )
        @printf("Generated tapered plate sample %d/%d -> C = %.3f\n", i, args.samples, compliance)
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

