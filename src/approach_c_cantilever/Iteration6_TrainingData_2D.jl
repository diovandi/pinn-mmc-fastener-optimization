using Printf
using Random
using Statistics
using DelimitedFiles
include("CantileverDiffFEA_2D.jl")

# ============================================================================
# Iteration 6 (2D): Training Data Generation for 2D Continuum Model
# Sample 50 random positions and compute compliance using 2D FEA
# ============================================================================

function generate_training_data_2d(n_samples=50, seed=42; nx=50, ny=5)
    """Generate training dataset with random support positions using 2D FEA."""
    
    Random.seed!(seed)
    data = []
    
    println("Generating $(n_samples) training samples using 2D continuum FEA...")
    println("Mesh: nx=$(nx), ny=$(ny)")
    println()
    
    for i in 1:n_samples
        # Random position in [0.1L, 0.9L]
        pos = 0.1*L_2D + rand() * 0.8*L_2D
        support_pos = [pos]
        
        # Compute compliance using 2D FEA
        c = solve_beam_compliance_2d(support_pos; nx=nx, ny=ny)
        
        push!(data, (pos, c))
        
        if i % 10 == 0
            @printf("  Sample %d/%d: pos=%.4f, C=%.6e\n", i, n_samples, pos, c)
        end
    end
    
    return data
end

function main()
    println("=" ^ 50)
    println("Iteration 6 (2D): Training Data Generation")
    println("Using 2D Continuum (Plane Stress) FEA")
    println("=" ^ 50)
    println()
    
    # Generate data with 2D FEA
    # Use coarser mesh for faster generation (can refine later)
    data = generate_training_data_2d(50; nx=30, ny=3)
    
    # Write CSV output
    output_file = "../../data/cantilever/iter_6_training_data_2d.csv"
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
    @printf("\nCompliance statistics (2D FEA):\n")
    @printf("  Min: %.6e J\n", minimum(compliances))
    @printf("  Max: %.6e J\n", maximum(compliances))
    @printf("  Mean: %.6e J\n", mean(compliances))
    @printf("  Std: %.6e J\n", std(compliances))
end

main()

