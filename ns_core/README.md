# ns_core

PyTorch library for the matrix sign map, the generalized Newton–Schulz
iteration and the spectral operators built from it.

- `matrices.py` — random test matrices: Gaussian, symmetric, semi-orthogonal
  factors, a matrix with an exactly prescribed spectrum, and rank-deficient /
  ill-conditioned matrices.
- `metrics.py` — reference SVD and eigendecomposition, reference spectral
  operators (`op_svd`, `op_eig`), spectral coordinates, frame residuals, and
  the Frobenius, relative Frobenius and spectral error norms.
- `ns_iteration.py` — the polynomials `bpoly_D` (coefficients, evaluation in
  the residual variable, orbits, `log10_error_orbit`, asymptotic error
  constant) and the matrix recursion (`ns_step_matrix`, `ns_orbit_matrix`).
  The orbit functions take a degree `D` or a coefficient tensor `coeffs`.
- `sign_map.py` — the matrix sign map: `sgn_svd` (exact), `make_sgn_ns` (the
  decomposition-free surrogate built from `ns_iteration`), both of type
  `msgn`, so any spectral operator in `cpwl.py` accepts either.
- `profiles.py` — the truncated series `B_D`, the basin radius `R_D`, the
  iteration count `K_D`, and the admissible quintics `x(1 + r1 t + r2 t^2)`:
  coefficients, `bmax`, slopes at 0 and 1, extremal slope, order of
  convergence and error constant.
- `cpwl.py` — the piecewise-linear scalar profiles and their sign forms on
  R^{m x n} (singular-value frame) and Sym^n (eigenvalue frame, built from
  the matrix absolute value).

## Conventions

- Functions that take a tensor create every internal tensor on its device and
  dtype. Functions without a tensor input (`bpoly_coeffs`,
  `truncated_series_coeffs`, the generators in `matrices.py`) take `device`
  and `dtype` keywords, defaulting to CPU / float64 or CPU / float32.
- Coefficients are computed in double precision and returned in the requested
  dtype.
- Randomized functions take an explicit `torch.Generator`.
- A quintic runs through the orbit functions by its coefficient tensor:
  `ns_orbit_matrix(M, None, 10, coeffs=quintic_coeffs(r1, r2))`.

## Running

From the repo root: `python -c "import ns_core"`, or
`from ns_core import matrices, ns_iteration, sign_map, profiles, cpwl`.
Each public function's docstring ends with a one-line usage.
