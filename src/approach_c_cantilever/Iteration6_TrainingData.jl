using Printf
using Random
using Statistics
using DelimitedFiles
include("CantileverDiffFEA.jl")

# ============================================================================
# Iteration 6: Training Data Generation
# Sample 50 random positions and compute compliance
# ============================================================================

function generate_training_data(n_samples=50, seed=42)
    """Generate training dataset with random support positions."""
    
    Random.seed!(seed)
    data = []
    
    println("Generating $(n_samples) training samples...")
    
    for i in 1:n_samples
        # Random position in [0.1L, 0.9L]
        pos = 0.1*L + rand() * 0.8*L
        support_pos = [pos]
        
        # Compute compliance
        c = solve_beam_compliance(support_pos)
        
        push!(data, (pos, c))
        
        if i % 10 == 0
            @printf("  Sample %d/%d: pos=%.4f, C=%.6e\n", i, n_samples, pos, c)
        end
    end
    
    return data
end

function main()
    println("=" ^ 50)
    println("Iteration 6: Training Data Generation")
    println("=" ^ 50)
    println()
    
    # Generate data
    data = generate_training_data(50)
    
    # Write CSV output
    output_file = "../../data/cantilever/iter_6_training_data.csv"
    open(output_file, "w") do f
        println(f, "support_pos,compliance,iteration")
        for (pos, c) in data
            @printf(f, "%.4f,%.6e,6\n", pos, c)
        end
    end
    
    println("\n✅ Generated $(length(data)) samples")
    println("✅ Results saved to $(output_file)")
    
    # Print statistics
    compliances = [d[2] for d in data]
    @printf("\nCompliance statistics:\n")
    @printf("  Min: %.6e J\n", minimum(compliances))
    @printf("  Max: %.6e J\n", maximum(compliances))
    @printf("  Mean: %.6e J\n", mean(compliances))
    @printf("  Std: %.6e J\n", std(compliances))
end

main()

