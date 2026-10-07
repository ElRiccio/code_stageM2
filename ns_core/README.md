# ns_core

PyTorch library for the matrix sign map, the Newton-Schulz iteration, and the
spectral operators built from them.

## Modules

- `matrices.py`: random test matrices (Gaussian, symmetric, prescribed
  spectrum, rank-deficient, ill-conditioned).
- `metrics.py`: SVD and eigendecomposition references, exact spectral operators
  (`op_svd`, `op_eig`), error norms.
- `ns_iteration.py`: the update polynomial, scalar orbits, and the matrix step
  (`ns_step_matrix`, and `ns_step_gram` which works on the Gram matrix).
  Orbit functions take a degree `D` or a `coeffs` tensor.
- `sign_map.py`: the sign map. `sgn_svd` is exact. `sgn_ns_fixed` runs K steps.
  `sgn_ns_until` runs until a residual tolerance. `make_sgn_ns_until` wraps it
  and records stats. All share one call signature (`msgn`).
- `profiles.py`: closed-form quantities per degree (truncated series, basin
  radius, step count, quintic coefficients).
- `orbit_tools.py`: test instance, orbit and errors for the convergence
  experiments.
- `plots.py`: plot helpers, each taking an optional axis.
- `timing.py`: `time_call` and `describe`.
- `cpwl.py`: piecewise-linear profiles and their versions on singular values
  and eigenvalues.

## Conventions

- Tensor inputs: internal tensors use the input's device and dtype. Functions
  without one take `device` and `dtype` keywords.
- Coefficients are computed in double precision, then cast.
- Random functions take a `torch.Generator`.
- A quintic is passed as `coeffs=quintic_coeffs(r1, r2)`.

Check the import with `python -c "import ns_core"`.
