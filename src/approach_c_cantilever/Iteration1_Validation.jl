using Printf
include("CantileverDiffFEA.jl")

# ============================================================================
# Iteration 1: Single Support Validation
# Compare Zygote gradients vs finite-difference
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
    println("Iteration 1: Single Support Validation")
    println("=" ^ 50)
    println()
    
    # Test support at x = 0.5
    support_pos = [0.5]
    
    println("Support Position: [$(support_pos[1])]")
    println()
    
    # Compute compliance
    c = solve_beam_compliance(support_pos)
    @printf("Compliance (FEA): %.6e J\n", c)
    println()
    
    # Compute gradients
    grad_zygote = ∇compliance_fea(support_pos)
    grad_fd = finite_difference_gradient(support_pos)
    
    @printf("Gradient (Zygote): [%.6e]\n", grad_zygote[1])
    @printf("Gradient (Finite Diff): [%.6e]\n", grad_fd[1])
    println()
    
    # Compute relative error
    rel_error = abs(grad_zygote[1] - grad_fd[1]) / (abs(grad_fd[1]) + 1e-12)
    @printf("Relative Error: %.2e", rel_error)
    
    if rel_error < 1e-6
        println(" ✓ PASS")
    else
        println(" ✗ FAIL")
    end
    
    # Write output file
    output_file = "../../data/cantilever/iter_1_validation.txt"
    open(output_file, "w") do f
        println(f, "Iteration 1: Single Support at x = 0.5")
        println(f, "─" ^ 50)
        @printf(f, "Support Position: [%.1f]\n", support_pos[1])
        @printf(f, "Compliance (FEA): %.6e J\n", c)
        @printf(f, "Gradient (Zygote): [%.6e]\n", grad_zygote[1])
        @printf(f, "Gradient (Finite Diff): [%.6e]\n", grad_fd[1])
        @printf(f, "Relative Error: %.2e", rel_error)
        if rel_error < 1e-6
            println(f, " ✓ PASS")
        else
            println(f, " ✗ FAIL")
        end
    end
    
    println("\n✅ Results saved to $(output_file)")
end

main()

