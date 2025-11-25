#!/usr/bin/env julia
"""
Helper script to append additional samples to existing dataset files.
Reads existing CSV, calculates how many more samples are needed, generates them,
and appends to the file.
"""

using Random
using CSV
using DataFrames
using Printf
using Dates

const ROOT_DIR = normpath(joinpath(@__DIR__, "../../.."))
const OUTPUT_ROOT = normpath(joinpath(ROOT_DIR, "../data/results/multi_geom_training"))

function append_lbracket_samples(output_path::String, target_samples::Int, load_case::Symbol, seed_offset::Int)
    APPROACH_A_DIR = normpath(joinpath(ROOT_DIR, "src/approach_a_pinn"))
    Base.include(Main, joinpath(APPROACH_A_DIR, "LBracketBenchmarkLogger.jl"))
    
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
    base_prob = load_problem()
    
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
            prob.coords, prob.elements, prob.fixed_dofs, loads,
            prob.centroids, prob.nu, prob.thick, prob.E_base, prob.E_bolt, prob.bolt_radius,
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
    
    prob = with_load_case(base_prob, load_case)
    new_rows = Vector{NamedTuple}(undef, needed)
    
    for i in 1:needed
        screws = random_layout(rng)
        compliance, safe = compliance_from_screws(screws, prob)
        new_rows[i] = (
            geometry = "l_bracket",
            load_case = String(load_case),
            sample_id = existing_samples + i,
            s1_x = safe[1],
            s1_y = safe[2],
            s2_x = safe[3],
            s2_y = safe[4],
            compliance = compliance,
            timestamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        )
        @printf("Generated L-bracket sample %d/%d -> C = %.3f\n", existing_samples + i, target_samples, compliance)
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

function append_tapered_samples(output_path::String, target_samples::Int, seed_offset::Int)
    FEA_DIR = normpath(joinpath(ROOT_DIR, "src/experiments/scenario_validation/fea"))
    Base.include(Main, joinpath(FEA_DIR, "NewPartBenchmark.jl"))
    
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
    
    ROW_A_X = 20.0
    ROW_B_X = 80.0
    ROW_Y_MIN = 12.0
    ROW_Y_MAX = 48.0
    ROW_SPACING = 15.0
    
    function enumerate_rows()
        y_vals = ROW_Y_MIN:ROW_SPACING:ROW_Y_MAX
        row_a = [(ROW_A_X, y) for y in y_vals]
        row_b = [(ROW_B_X, y) for y in y_vals]
        return row_a, row_b
    end
    
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

function append_channel_samples(output_path::String, target_samples::Int, load_case::Symbol, seed_offset::Int)
    FEA_DIR = normpath(joinpath(ROOT_DIR, "src/experiments/scenario_validation/fea"))
    Base.include(Main, joinpath(FEA_DIR, "RibbedChannelBenchmark.jl"))
    
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
    
    ROWS = [35.0, 80.0, 125.0]
    ROW_Y_RANGE = (8.0, 42.0)
    ROW_SPACING = 12.0
    
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

function main()
    if length(ARGS) < 3
        println("Usage: julia append_to_dataset.jl <geometry> <load_case> <target_samples> [output_path]")
        println("  geometry: l_bracket | tapered_plate | ribbed_channel")
        println("  load_case: horizontal_tip | vertical_tip | combined_tip | upward_tip | lateral_shear")
        exit(1)
    end
    
    geometry = ARGS[1]
    load_case_str = ARGS[2]
    target_samples = parse(Int, ARGS[3])
    output_path = length(ARGS) > 3 ? ARGS[4] : joinpath(OUTPUT_ROOT, "$(geometry)_$(load_case_str).csv")
    
    load_case = Symbol(load_case_str)
    seed_offset = 1000  # Base seed offset to avoid collisions
    
    if geometry == "l_bracket"
        append_lbracket_samples(output_path, target_samples, load_case, seed_offset)
    elseif geometry == "tapered_plate"
        append_tapered_samples(output_path, target_samples, seed_offset)
    elseif geometry == "ribbed_channel"
        append_channel_samples(output_path, target_samples, load_case, seed_offset)
    else
        error("Unknown geometry: $geometry")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

