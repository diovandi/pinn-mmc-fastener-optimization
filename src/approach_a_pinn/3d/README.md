# 3D PINN/FEA Framework

This directory contains the 3D extension of the differentiable FEA and PINN approach.

## Current Status

**Initial Structure Created** - Foundation for 3D topology optimization using tetrahedral elements.

## Files

- `DiffFEA_3D.jl` - 3D differentiable FEA solver skeleton
  - Placeholder for tetrahedral element formulation
  - Material matrix (D) for 3D elasticity implemented
  - Volume computation for tetrahedral elements
  - TODO: Complete B-matrix implementation and solver

## Implementation Plan

### Phase 1: Complete DiffFEA_3D.jl
1. Implement full B-matrix for linear tetrahedral elements
2. Complete sparse matrix assembly for 3D meshes
3. Add boundary condition handling for 3D geometries
4. Test with simple 3D geometry (cantilever beam, L-bracket)

### Phase 2: 3D Mesh Generation
1. Extend Gmsh pipeline for 3D volumetric meshing
2. Implement surface extraction and constraintable region tagging
3. Add mesh quality validation and refinement utilities
4. Create mesh visualization tools for 3D geometries

### Phase 3: 3D Dataset Generation
1. Implement screw position sampling in 3D space (x, y, z coordinates)
2. Add orientation DOFs for cylindrical fasteners (normal vector, rotation angle)
3. Extend differentiable FEA to handle 3D screw placement
4. Generate training datasets for 3D PINN surrogate

### Phase 4: Integration
1. Integrate 3D FEA with PINN training pipeline
2. Validate 3D results against commercial FEA (Ansys/CalculiX)
3. Compare 3D PINN vs 3D MMC performance
4. Document 3D extension in thesis chapters

## Dependencies

- Julia 1.8+
- Zygote.jl (for automatic differentiation)
- SparseArrays.jl
- LinearAlgebra.jl
- Gmsh.jl (for mesh generation - future)

## References

- 2D implementation: `../DiffFEA_2D.jl`
- 2D training: `../train_multi_geom.py`
- 3D geometry definitions: `../../../data/cad/3d/`

