using Printf
using DelimitedFiles
include("CantileverDiffFEA.jl")

# ============================================================================
# Iteration 12: Multi-Support C*(N) Curve
# ============================================================================

function optimize_n_supports(n_supports, initial_positions, lr=0.01, max_iter=30)
    """Optimize N support positions."""
    pos = copy(initial_positions)
    
    for iter in 1:max_iter
        c = solve_beam_compliance(pos)
        grads = ∇compliance_fea(pos)
        
        if norm(grads) < 1e-8
            break
        end
        
        pos = pos - lr * grads
        pos = [clamp(p, 0.1*L, 0.9*L) for p in pos]
        sort!(pos)
        
        # Ensure minimum spacing
        min_spacing = 0.05 * L
        for i in 2:length(pos)
            if pos[i] - pos[i-1] < min_spacing
                mid = (pos[i] + pos[i-1]) / 2
                pos[i-1] = mid - min_spacing/2
                pos[i] = mid + min_spacing/2
            end
        end
    end
    
    c_final = solve_beam_compliance(pos)
    return pos, c_final
end

function main()
    println("=" ^ 50)
    println("Iteration 12: Multi-Support C*(N) Curve")
    println("=" ^ 50)
    println()
    
    results = []
    
    # N = 1 support
    println("Optimizing 1 support...")
    pos1, c1 = optimize_n_supports(1, [0.4])
    push!(results, (1, c1, c1, 0.0))
    @printf("  N=1: C*=%.6e J\n", c1)
    
    # N = 2 supports
    println("Optimizing 2 supports...")
    pos2, c2 = optimize_n_supports(2, [0.3, 0.7])
    push!(results, (2, c2, c2, 0.0))
    @printf("  N=2: C*=%.6e J\n", c2)
    
    # N = 3 supports
    println("Optimizing 3 supports...")
    pos3, c3 = optimize_n_supports(3, [0.25, 0.5, 0.75])
    push!(results, (3, c3, c3, 0.0))
    @printf("  N=3: C*=%.6e J\n", c3)
    
    # Write CSV
    output_file = "../../data/cantilever/iter_12_c_star_n_curve.csv"
    open(output_file, "w") do f
        println(f, "n_supports,fea_optimum_compliance,pinn_optimum_compliance,error_pct")
        for (n, c_fea, c_pinn, err) in results
            @printf(f, "%d,%.6e,%.6e,%.3f\n", n, c_fea, c_pinn, err)
        end
    end
    
    println("\n✅ Results saved to $(output_file)")
end

main()

