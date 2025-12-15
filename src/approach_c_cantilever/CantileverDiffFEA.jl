# ============================================================================
# Cantilever Beam Differentiable FEA - Main Module
# ============================================================================
# This module provides both 1D (Euler-Bernoulli beam) and 2D (continuum)
# implementations for cantilever beam support placement optimization.

# Include both implementations
include("CantileverDiffFEA_1D.jl")
include("CantileverDiffFEA_2D.jl")

# Export main functions for backward compatibility
# 1D functions (default - backward compatible)
export solve_beam_compliance
export solve_beam_compliance_extended
export extract_tip_deflection
export ∇compliance_fea

# 1D specific exports
export solve_beam_compliance_1d
export solve_beam_compliance_extended_1d
export extract_tip_deflection_1d
export ∇compliance_fea_1d

# 2D specific exports
export solve_beam_compliance_2d
export solve_beam_compliance_extended_2d
export extract_tip_deflection_2d
export compute_von_mises_stress_2d
export ∇compliance_fea_2d
export compliance_2d
export grad_compliance_2d

# Constants
export E_1D, nu_1D, b_1D, h_1D, L_1D, q_1D, I_1D
export E_2D, nu_2D, b_2D, h_2D, L_2D, q_2D
