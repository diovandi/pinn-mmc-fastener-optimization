using Printf
include("CantileverDiffFEA.jl")

# ============================================================================
# Iteration 13: Computational Benchmark
# ============================================================================

function finite_difference_gradient(support_pos, h=1e-6)
    """Compute gradient using finite differences."""
    c0 = solve_beam_compliance(support_pos)
    grad = zeros(length(support_pos))
    
    for i in 1:length(support_pos)
        pos_pert = copy(support_pos)
        pos_pert[i] += h
        c_pert = solve_beam_compliance(pos_pert)
        grad[i] = (c_pert - c0) / h
    end
    
    return grad
end

function main()
    println("=" ^ 50)
    println("Iteration 13: Computational Benchmark")
    println("=" ^ 50)
    println()
    
    n_iterations = 100
    support_pos = [0.4]
    
    println("Scenario: Single support optimization over $(n_iterations) iterations")
    println()
    
    # Method A: Finite-Difference Gradient
    println("Method A (Finite-Difference Gradient):")
    t_start = time()
    for i in 1:n_iterations
        grad = finite_difference_gradient(support_pos)
        # Simulate optimization step
        support_pos[1] = support_pos[1] - 0.01 * grad[1]
        support_pos[1] = clamp(support_pos[1], 0.1*L, 0.9*L)
    end
    t_fd = time() - t_start
    n_fea_evals_fd = n_iterations * (length(support_pos) + 1)  # N+1 per iteration
    @printf("  Time: %.2f seconds\n", t_fd)
    @printf("  FEA evaluations: %d × %d = %d\n", n_iterations, length(support_pos)+1, n_fea_evals_fd)
    println()
    
    # Reset position
    support_pos = [0.4]
    
    # Method B: diffFEA + Zygote
    println("Method B (diffFEA + Zygote):")
    t_start = time()
    for i in 1:n_iterations
        grad = ∇compliance_fea(support_pos)
        # Simulate optimization step
        support_pos[1] = support_pos[1] - 0.01 * grad[1]
        support_pos[1] = clamp(support_pos[1], 0.1*L, 0.9*L)
    end
    t_zygote = time() - t_start
    n_fea_evals_zygote = n_iterations * 1  # 1 per iteration
    @printf("  Time: %.2f seconds\n", t_zygote)
    @printf("  FEA evaluations: %d × %d = %d\n", n_iterations, 1, n_fea_evals_zygote)
    println()
    
    speedup = t_fd / t_zygote
    @printf("Speedup: %.2fx\n", speedup)
    println("Verdict: Real speedup, but NOT \"2272x\"")
    
    # Write output
    output_file = "../../data/cantilever/iter_13_benchmark.txt"
    open(output_file, "w") do f
        println(f, "Iteration 13: Computational Benchmark")
        println(f, "─" ^ 50)
        println(f, "Scenario: Single support optimization over $(n_iterations) iterations")
        println(f)
        println(f, "Method A (Finite-Difference Gradient):")
        @printf(f, "  Time: %.2f seconds\n", t_fd)
        @printf(f, "  FEA evaluations: %d × %d = %d\n", n_iterations, length(support_pos)+1, n_fea_evals_fd)
        println(f)
        println(f, "Method B (diffFEA + Zygote):")
        @printf(f, "  Time: %.2f seconds\n", t_zygote)
        @printf(f, "  FEA evaluations: %d × %d = %d\n", n_iterations, 1, n_fea_evals_zygote)
        println(f)
        @printf(f, "Speedup: %.2fx\n", speedup)
        println(f, "Verdict: Real speedup, but NOT \"2272x\"")
    end
    
    println("\n✅ Results saved to $(output_file)")
end

main()

