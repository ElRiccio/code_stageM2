"""ns_core: PyTorch library for the matrix sign map and the generalized
Newton-Schulz iteration.

Modules
-------
matrices      Random test matrices: Gaussian, symmetric, semi-orthogonal
              factors, prescribed spectrum, rank-deficient and ill-conditioned.
metrics       Reference SVD / eigendecomposition, reference spectral operators,
              spectral coordinates, frame residuals and error norms.
ns_iteration  The polynomials bpoly_D and the Newton-Schulz recursion they
              drive, on scalars and on matrices.
sign_map      The matrix sign map msgn: exact (SVD) and the decomposition-free
              surrogate built from ns_iteration.
profiles      The truncated series B_D, the basin radius R_D, the iteration
              count K_D, and the admissible quintics given by (r1, r2).
cpwl          Continuous piecewise-linear scalar profiles and their sign
              forms on R^{m x n} and Sym^n.
"""
