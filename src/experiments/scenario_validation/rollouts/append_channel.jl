#!/usr/bin/env julia

using Random
using CSV
using DataFrames
using Printf
using Dates

const ROOT_DIR = normpath(joinpath(@__DIR__, "../../.."))
const FEA_DIR = normpath(joinpath(@__DIR__, "../fea"))
const OUTPUT_ROOT = normpath(joinpath(ROOT_DIR, "../data/results/multi_geom_training"))

include(joinpath(FEA_DIR, "RibbedChannelBenchmark.jl"))

const ROWS = [35.0, 80.0, 125.0]
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

function main()
    if length(ARGS) < 3
        println("Usage: julia append_channel.jl <load_case> <target_samples> <output_path> [seed_offset]")
        exit(1)
    end
    
    load_case = Symbol(ARGS[1])
    target_samples = parse(Int, ARGS[2])
    output_path = ARGS[3]
    seed_offset = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 3000
    
    # Read existing data if file exists
    existing_samples = 0
    if isfile(output_path)
        df_existing = CSV.read(output_path, DataFrame)
        existing_samples = nrow(df_existing)
        println("Found $existing_samples existing samples in $output_path")
    else
        println("Creating new file: $output_path")
    end
    
    needed = target_samples - existing_samples
    if needed <= 0
        println("✅ Already have $existing_samples samples (target: $target_samples). No generation needed.")
        return
    end
    
    println("Generating $needed additional samples...")
    
    rng = MersenneTwister(seed_offset + existing_samples)
    mesh = build_channel_mesh()
    positions = enumerate_positions()
    
    new_rows = Vector{NamedTuple}(undef, needed)
    for i in 1:needed
        layout = sample_layout(rng, positions)
        compliance = solve_compliance(mesh, layout, load_case)
        new_rows[i] = (
            geometry = "ribbed_channel",
            load_case = String(load_case),
            sample_id = existing_samples + i,
            s1_x = layout[1],
            s1_y = layout[2],
            s2_x = layout[3],
            s2_y = layout[4],
            s3_x = layout[5],
            s3_y = layout[6],
            compliance = compliance,
            timestamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        )
        @printf("Generated ribbed channel sample %d/%d -> C = %.3f\n", existing_samples + i, target_samples, compliance)
    end
    
    new_df = DataFrame(new_rows)
    
    if existing_samples > 0
        df_existing = CSV.read(output_path, DataFrame)
        combined_df = vcat(df_existing, new_df)
    else
        combined_df = new_df
    end
    
    CSV.write(output_path, combined_df)
    println("✅ Appended $needed samples. Total: $(nrow(combined_df)) samples in $output_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

