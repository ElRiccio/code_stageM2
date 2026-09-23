# ns_core

The reusable numerical library. No module here saves a file or reads a
command-line argument — that belongs to `experiments`.

- `matrices.py` — random test matrices: Gaussian, symmetric, semi-orthogonal
  factors, a matrix with an exactly prescribed spectrum, and
  rank-deficient/ill-conditioned constructions (built by an explicit SVD
  overwrite, kept distinct from the decomposition-free algorithms these
  matrices are used to test).
- `metrics.py` — reference SVD/eigendecomposition, reference spectral
  operators (`op_svd`, `op_eig`, the decomposition-based ground truth),
  spectral coordinates and frame residuals, and the error norms
  (Frobenius, relative Frobenius, spectral) used throughout.
- `ns_iteration.py` — the degree-(2D+1) polynomial family p_D, both scalar
  (orbits, the deflated error orbit) and matrix (the NS recursion via the
  odd matrix polynomial Phi).
- `sign_map.py` — the matrix sign map: `sgn_svd` (exact reference) and
  `make_sgn_ns` (the decomposition-free surrogate built from
  `ns_iteration`), unified behind a single `Sgn` callable type so any
  spectral operator in `cpwl.py` can be evaluated with either.
- `profiles.py` — admissible-profile constants (mu_D, rho_D, lambda_D),
  the burn-in bound, and the basin-of-attraction radius / iteration-count
  bound built from the truncated series B_D.
- `cpwl.py` — the six piecewise-linear scalar profiles and their
  decomposition-free sign forms, on both R^{m x n} (via the singular-value
  frame) and Sym^n (via the eigenvalue frame, built from the matrix
  absolute value).

## Conventions

- Every matrix-generating or randomized function takes an explicit
  `torch.Generator`; nothing relies on global RNG state.
- `device`/`dtype` are explicit keyword arguments, defaulting to CPU /
  float32. Coefficient computations that are reused across many steps
  (`bpoly_coeffs`, `deflation_coeffs`) are done in float64 regardless of the
  iteration's own dtype, since their own rounding error should not be the
  bottleneck.
- Functions operate on a single matrix; repeating/averaging over several
  random matrices is done by looping at the experiment level, not by
  batching inside `ns_core`.
- Docstrings describe the underlying formula directly rather than citing a
  thesis theorem/definition number.
