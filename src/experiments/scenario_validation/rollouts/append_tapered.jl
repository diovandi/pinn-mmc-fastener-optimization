#!/usr/bin/env julia

using Random
using CSV
using DataFrames
using Printf
using Dates

const ROOT_DIR = normpath(joinpath(@__DIR__, "../../.."))
const FEA_DIR = normpath(joinpath(@__DIR__, "../fea"))
const OUTPUT_ROOT = normpath(joinpath(ROOT_DIR, "../data/results/multi_geom_training"))

include(joinpath(FEA_DIR, "NewPartBenchmark.jl"))

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

function main()
    if length(ARGS) < 2
        println("Usage: julia append_tapered.jl <target_samples> <output_path> [seed_offset]")
        exit(1)
    end
    
    target_samples = parse(Int, ARGS[1])
    output_path = ARGS[2]
    seed_offset = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 2000
    
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
    mesh = build_plate_mesh()
    row_a, row_b = enumerate_rows()
    
    new_rows = Vector{NamedTuple}(undef, needed)
    for i in 1:needed
        s1 = row_a[rand(rng, 1:length(row_a))]
        s2 = row_b[rand(rng, 1:length(row_b))]
        layout = clamp_screws([s1[1], s1[2], s2[1], s2[2]])
        compliance = solve_compliance(mesh, layout)
        new_rows[i] = (
            geometry = "tapered_plate",
            load_case = "combined_tip",
            sample_id = existing_samples + i,
            s1_x = layout[1],
            s1_y = layout[2],
            s2_x = layout[3],
            s2_y = layout[4],
            compliance = compliance,
            timestamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        )
        @printf("Generated tapered plate sample %d/%d -> C = %.3f\n", existing_samples + i, target_samples, compliance)
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

