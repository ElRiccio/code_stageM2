# ns_core

A small PyTorch library for the matrix sign map, the Newton-Schulz iteration,
and the spectral operators you can build from them.

## Modules

- `matrices.py` makes random test matrices: Gaussian, symmetric, one with an
  exactly chosen spectrum, and rank-deficient or ill-conditioned ones.
- `metrics.py` holds the ground truth: SVD, eigendecomposition, the exact
  spectral operators (`op_svd`, `op_eig`), and the error norms (Frobenius,
  relative Frobenius, spectral).
- `ns_iteration.py` is the iteration itself: the polynomial p_D (coefficients
  and evaluation), scalar orbits, and the matrix step. `ns_step_matrix` is the
  plain version; `ns_step_gram` gives the same result more cheaply by working
  with the Gram matrix on the smaller side, and can reuse one you've already
  computed (`gram_matrix`). Orbit functions take either a degree `D` or a
  `coeffs` tensor for a custom polynomial.
- `sign_map.py` is the matrix sign map. `sgn_svd` is exact; `make_sgn_ns` is
  the SVD-free approximation. Both have the same shape (matrix in, sign out,
  called `msgn`), so anything in `cpwl.py` accepts either. The Newton-Schulz
  versions come in three flavors: `sgn_ns_fixed` runs K steps,
  `sgn_ns_until` runs until the residual `||X (X^T X - I)||_F` reaches a target
  (it is zero at singular values 0 and 1, so rank-deficient and zero matrices
  are fine), and
  `make_sgn_ns_until` wraps that as an `msgn`. Give it a `stats` list and it
  records (steps, reached) for every internal call.
- `profiles.py` has the closed-form quantities: the truncated series B_D, the
  basin radius R_D, the step count K_D, and the admissible quintics
  (coefficients, slopes, order and constant of convergence).
- `orbit_tools.py` supports the convergence experiments: building a test
  matrix with log-spaced singular values plus its exact sign, running the
  iteration, and measuring errors, ranks and the first step below a target.
  `run_orbit` freezes the iterate once its error reaches `eps`, so noise in the
  null space of a rank-deficient matrix is not amplified afterwards. It takes
  either a degree or explicit `coeffs`, so any odd polynomial can be run.
- `plots.py` has the plotting helpers, all taking an optional axis: error
  curves, the e_{k+1} vs e_k map, rank curves, spectra, and the timing plots
  (time vs n, vs smin, vs degree, and `plot_step_cost_vs_degree` for the time of one step). The time-vs-degree plot draws the exact-SVD
  time as a dashed line when it was timed (`show_svd=False` hides it);
  `add_svd_reference_line` does the same on any axis.
- `timing.py` has `time_call` (one timed call, with warm-up and CUDA sync) and
  `describe` (median, mean, std of repeated timings).
- `cpwl.py` has the piecewise-linear profiles (clip, soft threshold, leaky
  ReLU, ...) and their matrix versions: rectangular ones acting on singular
  values, symmetric ones acting on eigenvalues (built from the matrix absolute
  value).

## Conventions

- A function that takes a tensor builds its internal tensors on that tensor's
  device and dtype. One without a tensor input (`bpoly_coeffs`,
  `truncated_series_coeffs`, the generators in `matrices.py`) takes `device`
  and `dtype` keywords, defaulting to CPU with float64 or float32.
- Coefficients are computed in double precision and cast to the requested
  dtype at the end.
- Anything random takes an explicit `torch.Generator`. Nothing uses global
  seeding.
- A quintic goes through the orbit functions as a coefficient tensor:
  `ns_orbit_matrix(M, None, 10, coeffs=quintic_coeffs(r1, r2))`.

## Docstrings

Each function's docstring lists its inputs, one short line each, then what it
returns, then at most one `Note:` for anything that would surprise a caller.
The reasoning behind each formula lives in the thesis, not here.

## Trying it

From the repo root:

```
python -c "import ns_core"
```

or `from ns_core import matrices, ns_iteration, sign_map, profiles, cpwl`.
