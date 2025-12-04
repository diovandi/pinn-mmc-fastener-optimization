using Printf
include("CantileverDiffFEA.jl")

# ============================================================================
# Iteration 5: Two Supports with Varying Positions
# ============================================================================

function optimize_two_supports(initial_pos=[0.30, 0.70], lr=0.01, max_iter=30, tol=1e-8)
    """Optimize two support positions using gradient descent."""
    
    pos = copy(initial_pos)
    
    println("Initial positions: [$(pos[1]), $(pos[2])]")
    
    for iter in 1:max_iter
        # Compute compliance and gradient
        c = solve_beam_compliance(pos)
        grads = ∇compliance_fea(pos)
        
        # Print progress
        if iter <= 5 || iter % 5 == 0
            @printf("Iter %2d: pos=[%.4f, %.4f], C=%.6e, grad=[%.6e, %.6e]\n", 
                    iter, pos[1], pos[2], c, grads[1], grads[2])
        end
        
        # Check convergence
        if norm(grads) < tol
            println("Converged at iteration $iter")
            break
        end
        
        # Gradient descent step
        pos = pos - lr * grads
        
        # Clamp to valid range [0.1L, 0.9L] and ensure ordering
        pos = [clamp(p, 0.1*L, 0.9*L) for p in pos]
        sort!(pos)  # Ensure pos[1] < pos[2]
        
        # Prevent supports from getting too close
        min_spacing = 0.05 * L
        if pos[2] - pos[1] < min_spacing
            mid = (pos[1] + pos[2]) / 2
            pos[1] = mid - min_spacing/2
            pos[2] = mid + min_spacing/2
        end
    end
    
    # Compute final compliance
    c_final = solve_beam_compliance(pos)
    return pos, c_final
end

function main()
    println("=" ^ 50)
    println("Iteration 5: Two Supports")
    println("=" ^ 50)
    println()
    
    # Initial positions
    initial_pos = [0.30, 0.70]
    c_init = solve_beam_compliance(initial_pos)
    
    println("Initial Configuration:")
    println("  Support Positions: [$(initial_pos[1]), $(initial_pos[2])]")
    @printf("  Compliance (FEA): %.6e J\n", c_init)
    println()
    
    # Optimize
    opt_pos, c_opt = optimize_two_supports(initial_pos)
    
    println()
    println("Optimized Configuration:")
    println("  Support Positions: [$(opt_pos[1]), $(opt_pos[2])]")
    @printf("  Compliance: %.6e J\n", c_opt)
    
    # Compute gradient at optimum
    grads = ∇compliance_fea(opt_pos)
    println("  Gradient: [$(grads[1]), $(grads[2])]")
    
    # Write output file
    output_file = "../../data/cantilever/iter_5_two_supports.txt"
    open(output_file, "w") do f
        println(f, "Iteration 5: Two Supports")
        println(f, "─" ^ 50)
        println(f, "Support Positions: [$(initial_pos[1]), $(initial_pos[2])]")
        @printf(f, "Compliance (FEA): %.6e J\n", c_init)
        @printf(f, "Gradient: [%.6e, %.6e]\n", ∇compliance_fea(initial_pos)...)
        println(f)
        println(f, "Optimized Positions: [$(opt_pos[1]), $(opt_pos[2])]")
        @printf(f, "Compliance: %.6e J\n", c_opt)
    end
    
    println("\n✅ Results saved to $(output_file)")
end

main()

