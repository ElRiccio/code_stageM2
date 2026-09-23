# code_stageM2

Numerical experiments for the Master's thesis chapter on the matrix sign
map, the generalized Newton–Schulz iteration, and the piecewise-linear
spectral operators built from it. Assumes the reader has already read the
relevant thesis chapters; nothing here re-derives the theory.

## Layout

- `ns_core/` — reusable PyTorch library: matrix generation, reference
  SVD/eigendecomposition, error norms, the Newton–Schulz iteration, the
  matrix sign map, admissible-profile constants, and the CPWL catalogue.
  See `ns_core/README.md`.
- `experiments/` — one module per numerical question, built on `ns_core`.
  Experiment functions return raw results; plotting is separate, so results
  can be replotted without rerunning. See `experiments/README.md`.
- `legacy_numpy/` — the original NumPy implementation and experiment
  scripts, kept as reference during the PyTorch migration. Not imported
  from anywhere in `ns_core` or `experiments`.

## Stack

Python, PyTorch (CPU and GPU), NumPy, Matplotlib. Dependencies are pinned
in `requirements.txt`.

## Status

`ns_core` is implemented: matrix generation, reference SVD/eigendecomposition
and error norms, the Newton-Schulz iteration (scalar and matrix readings),
the matrix sign map (exact and NS-surrogate), admissible-profile constants,
and the CPWL catalogue with its rectangular and symmetric sign forms. Cross-
checked against `legacy_numpy/ns_utils.py` on representative inputs
(coefficients, `bpoly_eval`, admissible constants, basin radius, `Sgn`,
every CPWL sign form, `op_svd`/`op_eig`, burn-in and iteration-count bounds)
during development; no persisted test suite, per the project's convention.

`experiments/plotting.py` (generic spectrum/error-decay plot helpers) is
implemented. No experiment modules exist yet — that's the next pass, one
module per reviewer-comment question (see `experiments/README.md`).
