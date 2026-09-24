# code_stageM2

Numerical code for the Master's thesis chapter on the matrix sign map, the
generalized Newton–Schulz iteration, and the piecewise-linear spectral
operators built from it. Assumes the reader has already read the relevant
thesis chapters; nothing here re-derives the theory.

## Layout

- `ns_core/` — reusable PyTorch library: matrix generation, reference
  SVD/eigendecomposition, error norms, the Newton–Schulz iteration, the
  matrix sign map, admissible-profile constants, and the CPWL catalogue.
  See `ns_core/README.md`.
- `experiments/` — one module per convergence experiment (config dataclass
  and `run_*` function): `convergence.py`, `conditioning.py`,
  `rank_deficiency.py`. Uses `ns_core`.
- `notebooks/` — one notebook per experiment (`exp1_convergence`,
  `exp2_conditioning`, `exp3_rank_deficiency`): edit the config, run, read
  the plots. Start Jupyter from `notebooks/` or the repo root.
- `old_numpy/` — the original NumPy implementation and its numbered
  experiment scripts (figures for the thesis). Standalone; independent of
  `ns_core`. See `old_numpy/README.md`.

## Setup

Python, PyTorch, NumPy, Matplotlib. Install with
`pip install -r requirements.txt`, then run everything from the repo root.
