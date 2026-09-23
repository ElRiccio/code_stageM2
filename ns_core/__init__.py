"""ns_core: reusable PyTorch library for the matrix sign / Newton-Schulz thesis.

Modules
-------
matrices    Random test matrices: Gaussian, orthogonal factors, rank-deficient
            and conditioning-controlled constructions.
metrics     Error norms, reference SVD/eigendecomposition, reference spectral
            operators (the decomposition-based ground truth), spectral
            coordinates and frame residuals.
ns_iteration  The degree-D polynomial family and the Newton-Schulz recursion
            it drives, on both the scalar and the matrix reading.
sign_map    The matrix sign map itself: the exact (SVD-based) reference and
            the decomposition-free surrogate built from ns_iteration.
profiles    Admissible-profile constants derived from the truncated series
            B_D: the basin-of-attraction radius and the iteration-count
            bound.
cpwl        The catalogue of continuous piecewise-linear scalar profiles and
            their decomposition-free sign forms, on Sym^n and on R^{m x n}.

None of these modules read command-line arguments or write files; that is
left to the `experiments` package.
"""
