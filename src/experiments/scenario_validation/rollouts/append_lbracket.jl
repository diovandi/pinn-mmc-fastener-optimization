#!/usr/bin/env julia

using Random
using CSV
using DataFrames
using Printf
using Dates

const ROOT_DIR = normpath(joinpath(@__DIR__, "../../.."))
const APPROACH_A_DIR = normpath(joinpath(@__DIR__, "../../../approach_a_pinn"))
const OUTPUT_ROOT = normpath(joinpath(ROOT_DIR, "../data/results/multi_geom_training"))

include(joinpath(APPROACH_A_DIR, "LBracketBenchmarkLogger.jl"))

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

function main()
    if length(ARGS) < 3
        println("Usage: julia append_lbracket.jl <load_case> <target_samples> <output_path> [seed_offset]")
        exit(1)
    end
    
    load_case = Symbol(ARGS[1])
    target_samples = parse(Int, ARGS[2])
    output_path = ARGS[3]
    seed_offset = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 1000
    
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

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

