#!/usr/bin/env julia

using Printf
using DelimitedFiles

include("CantileverDiffFEA.jl")

const OUTPUT_CSV = "../../data/cantilever/iter_2-4_optimization_2d.csv"
const OUTPUT_SUMMARY = "../../data/cantilever/iter_2-4_optimization_2d_summary.txt"

"""
Run gradient-based optimization of a single intermediate support position
for the 2D continuum cantilever beam, mirroring Iterations 2–4 (1D case).
"""
function main()
    # Optimization hyperparameters
    max_iters = 40
    α = 0.15             # step size
    x = 0.30             # initial support position (m)
    nx = 40
    ny = 6

    header = ["iteration", "support_pos", "compliance", "gradient"]
    rows = Float64[]

    println("Iteration 2–4 (2D): Single-support optimization with 2D diffFEA")
    println("---------------------------------------------------------------")
    @printf("Initial support position: %.4f m\n\n", x)

    for k in 1:max_iters
        C = compliance_2d(x; nx=nx, ny=ny)
        g = grad_compliance_2d(x; nx=nx, ny=ny)

        @printf("Iter %2d: x = %.4f m, C = %10.6e J, dC/dx = %10.6e\n",
                k, x, C, g)

        append!(rows, [k, x, C, g])

        # Gradient descent update
        x_new = x - α * g
        # Project to interior (avoid exactly at clamps)
        x = clamp(x_new, 0.1, 0.9)
    end

    # Reshape rows into N×4 matrix
    data = reshape(rows, 4, :)'
    open(OUTPUT_CSV, "w") do io
        writedlm(io, header, ',')
        writedlm(io, data, ',')
    end

    # Summary
    final_iter = size(data, 1)
    final_x = data[end, 2]
    final_C = data[end, 3]

    open(OUTPUT_SUMMARY, "w") do io
        println(io, "Iteration 2–4 (2D): Optimization Summary")
        println(io, "-----------------------------------------")
        @printf(io, "Final iteration: %d\n", final_iter)
        @printf(io, "Optimal support position (approx): %.6f m\n", final_x)
        @printf(io, "Optimal compliance (approx):       %.6e J\n", final_C)
    end

    println("\nSaved 2D optimization log to: $OUTPUT_CSV")
    println("Saved 2D optimization summary to: $OUTPUT_SUMMARY")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

using Printf
include("CantileverDiffFEA.jl")

# ============================================================================
# Iterations 2-4 (2D): Single Support Optimization with diffFEA continuum model
# ============================================================================

function optimize_single_support_2d(initial_pos=0.30;
                                    lr=5e-3,
                                    max_iter=20,
                                    tol=1e-6,
                                    nx=40,
                                    ny=6)
    """
    Gradient-descent optimization of support position for the 2D continuum model.
    """
    results = Tuple{Int,Float64,Float64,Float64}[]
    pos = initial_pos

    for iter in 1:max_iter
        c = compliance_2d(pos; nx=nx, ny=ny)
        grad = grad_compliance_2d(pos; nx=nx, ny=ny)
        push!(results, (iter, pos, c, grad))

        if abs(grad) < tol
            println("✅ Converged (|grad| < tol) at iteration $iter")
            break
        end

        # gradient descent step
        pos -= lr * grad
        pos = clamp(pos, 0.15 * L_2D, 0.85 * L_2D)

        if iter <= 4 || iter % 5 == 0
            @printf("Iter %2d: pos=%.4f m | C=%.6e J | grad=%.6e\n", iter, pos, c, grad)
        end
    end

    return results
end

function write_results(results)
    data_dir = joinpath(@__DIR__, "..", "..", "data", "cantilever")
    csv_path = joinpath(data_dir, "iter_2-4_optimization_2d.csv")
    summary_path = joinpath(data_dir, "iter_2-4_optimization_2d_summary.txt")

    open(csv_path, "w") do io
        println(io, "iteration,support_pos,compliance,gradient")
        for (iter, pos, c, grad) in results
            @printf(io, "%d,%.6f,%.6e,%.6e\n", iter, pos, c, grad)
        end
    end

    final_iter, final_pos, final_c, _ = results[end]
    open(summary_path, "w") do io
        println(io, "2D diffFEA Optimization Summary")
        println(io, "--------------------------------")
        println(io, "Iterations run: ", final_iter)
        @printf(io, "Final support position: %.6f m\n", final_pos)
        @printf(io, "Final compliance: %.6e J\n", final_c)
    end

    println("✅ Saved CSV -> $csv_path")
    println("✅ Saved summary -> $summary_path")
end

function main()
    println("="^60)
    println("Iterations 2-4 (2D): Single Support Optimization")
    println("="^60)
    println()

    results = optimize_single_support_2d()
    write_results(results)
end

main()


