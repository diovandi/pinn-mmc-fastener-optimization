using LinearAlgebra
using Zygote
using SparseArrays

# --- 1. Differentiable FEA Solver ---
# Solves K * u = f for u, then returns compliance = u' * f
# Note: We use a dense solver for simplicity (Zygote handles dense linear solves well)
# For 3D systems with 10k+ DOF, we will need a custom adjoint, but this works for prototypes.

function solve_compliance(stiffness_values::Vector{T}, forces::Vector{T}) where T
    # In a real FEA, we assemble K from element stiffness matrices.
    # Here, for the "Toy Problem", we assume a simplified spring system.
    # K is a diagonal matrix where diagonal entries are the stiffnesses.
    
    # Create diagonal stiffness matrix (simulating structural elements)
    # We add a small epsilon to avoid singularity if stiffness goes to 0
    K = Diagonal(stiffness_values .+ 1e-6)
    
    # Solve equilibrium: K * u = f
    # The "\" operator is differentiable in Julia!
    u = K \ forces
    
    # Compute Compliance (Strain Energy * 2)
    # C = u' * K * u = u' * f
    compliance = dot(u, forces)
    
    return compliance
end

# --- 2. Main Test Loop ---
function main()
    println("--- Differentiable FEA Test (Julia) ---")
    
    # Define a simple system: 3 elements (springs)
    # We want to find which spring needs to be stiffest to support the load.
    initial_stiffness = [1.0, 1.0, 1.0]
    loads = [0.0, 10.0, 0.0] # Load applied only on the middle node
    
    println("Initial Stiffness: ", initial_stiffness)
    println("Applied Loads: ", loads)
    
    # 1. Forward Pass
    c = solve_compliance(initial_stiffness, loads)
    println("Initial Compliance: ", c)
    
    # 2. Backward Pass (Sensitivity Analysis)
    # We ask: "How does Compliance change if we change the stiffness of each element?"
    grads = Zygote.gradient(s -> solve_compliance(s, loads), initial_stiffness)[1]
    
    println("\n--- Sensitivity Results ---")
    println("Gradient (dC/dk): ", grads)
    
    # Interpretation
    println("\nInterpretation:")
    println("Element 2 Gradient is: ", grads[2])
    if grads[2] < grads[1] && grads[2] < grads[3]
        println("✅ CORRECT: The gradient is most negative for Element 2.")
        println("   This means increasing stiffness at Element 2 reduces compliance the most.")
        println("   (This makes sense because the load is at Node 2!)")
    else
        println("❌ SOMETHING IS WRONG.")
    end
end

main()
