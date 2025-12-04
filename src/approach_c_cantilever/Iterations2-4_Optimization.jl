using Printf
using DelimitedFiles
include("CantileverDiffFEA.jl")

# ============================================================================
# Iterations 2-4: Single Support Optimization
# Gradient descent optimization loop
# ============================================================================

function optimize_single_support(initial_pos=0.3, lr=0.01, max_iter=20, tol=1e-8)
    """Optimize single support position using gradient descent."""
    
    results = []
    pos = initial_pos
    
    for iter in 1:max_iter
        # Compute compliance and gradient
        support_pos = [pos]
        c = solve_beam_compliance(support_pos)
        grad = ∇compliance_fea(support_pos)[1]
        
        # Record
        push!(results, (iter, pos, c, grad))
        
        # Check convergence
        if abs(grad) < tol
            println("Converged at iteration $iter")
            break
        end
        
        # Gradient descent step
        pos = pos - lr * grad
        
        # Clamp to valid range [0.1L, 0.9L]
        pos = clamp(pos, 0.1*L, 0.9*L)
        
        # Print progress
        if iter <= 4 || iter % 5 == 0
            @printf("Iter %2d: pos=%.4f, C=%.6e, grad=%.6e\n", iter, pos, c, grad)
        end
    end
    
    return results
end

function main()
    println("=" ^ 50)
    println("Iterations 2-4: Single Support Optimization")
    println("=" ^ 50)
    println()
    
    # Run optimization
    results = optimize_single_support(0.3, 0.01, 20)
    
    # Write CSV output
    output_file = "../../data/cantilever/iter_2-4_optimization.csv"
    open(output_file, "w") do f
        println(f, "iteration,position,compliance,gradient")
        for (iter, pos, c, grad) in results
            @printf(f, "%d,%.4f,%.6e,%.6e\n", iter, pos, c, grad)
        end
    end
    
    println("\n✅ Results saved to $(output_file)")
    println("\nFinal optimized position: $(results[end][2])")
    @printf("Final compliance: %.6e J\n", results[end][3])
end

main()

