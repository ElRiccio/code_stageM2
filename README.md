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

Currently: `ns_core` and `experiments` are scaffolded (module layout,
function signatures, docstrings) but not yet implemented — this is the
shared-plumbing pass addressing the reviewer's request for an explicit
experimental protocol (test matrices, rank-deficiency construction,
reference computations, error norms) before the individual experiments are
built on top of it.
