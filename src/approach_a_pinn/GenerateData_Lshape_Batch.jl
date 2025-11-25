#!/usr/bin/env julia

using LinearAlgebra
using Printf
using Random
using DelimitedFiles
using Zygote

include("LBracketBenchmarkLogger.jl")

const OUTPUT_FILE = joinpath(@__DIR__, "../../data/results/pinn_training_data.csv")
const DEFAULT_TARGET = 90

function read_existing(path)
    if !isfile(path)
        return zeros(Float64, 0, 5)
    end
    data = readdlm(path, ',', Float64)
    if ndims(data) == 1
        data = reshape(data, 1, :)
    end
    return data
end

function random_initial()
    screws = zeros(4)
    for s in 1:2
        if rand() > 0.5
            screws[2*s-1] = rand() * 25.0
            screws[2*s] = rand() * 100.0
        else
            screws[2*s-1] = rand() * 100.0
            screws[2*s] = rand() * 25.0
        end
    end
    return screws
end

function generate_sample(prob::LBracketProblem; steps=12)
    screws = enforce_lbracket_bounds(random_initial())
    for _ in 1:steps
        loss_fn = s -> begin
            c, _ = compliance_from_screws(s, prob)
            return c
        end
        val, grads = Zygote.withgradient(loss_fn, screws)
        g = grads[1]
        if !isfinite(val) || any(!isfinite, g)
            error("non-finite")
        end
        g = clamp.(g, -80.0, 80.0)
        screws .-= 2.0 .* g ./ (norm(g) + 1e-8)
        screws = enforce_lbracket_bounds(screws)
    end
    final_c, safe = compliance_from_screws(screws, prob)
    return safe, final_c
end

function append_rows(path, rows)
    open(path, isfile(path) ? "a" : "w") do io
        for row in rows
            write(io, join(row, ","), "\n")
        end
    end
end

function main()
    prob = load_problem()
    existing = read_existing(OUTPUT_FILE)
    existing_count = size(existing, 1)
    target_total = parse(Int, get(ENV, "PINN_TARGET_SAMPLES", string(DEFAULT_TARGET)))

    if existing_count >= target_total
        println("Dataset already has $existing_count samples (>= $target_total)")
        return
    end

    needed = target_total - existing_count
    println("Generating $needed additional samples (current $existing_count, target $target_total)")
    ensure_dirs()

    new_rows = Vector{Vector{Float64}}()
    attempts = 0
    while length(new_rows) < needed && attempts < needed * 6
        attempts += 1
        try
            screws, comp = generate_sample(prob)
            push!(new_rows, [screws[1], screws[2], screws[3], screws[4], comp])
            @printf("✅ Sample %d/%d -> C=%.2f\n", length(new_rows), needed, comp)
        catch err
            @printf("⚠️ Attempt %d failed: %s\n", attempts, err)
        end
    end

    if length(new_rows) < needed
        @warn "Requested $needed samples but only generated $(length(new_rows))"
    end

    append_rows(OUTPUT_FILE, new_rows)
    println("✅ Added $(length(new_rows)) samples. New total = $(existing_count + length(new_rows))")
end

main()

