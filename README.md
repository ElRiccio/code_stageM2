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
  `rank_deficiency.py`, the timing comparison with the SVD
  `svd_timing.py`, the time needed to reach a target accuracy
  `time_to_accuracy.py`, and the CPWL spectral operator built on top of the
  matrix sign map `cpwl_operator.py`. `cpwl_svd_timing.py` and
  `cpwl_time_to_accuracy.py` re-run `svd_timing.py` and `time_to_accuracy.py`
  on that CPWL operator instead of the raw sign map, with every internal
  sign call stopped by tolerance rather than a fixed iteration count. Uses
  `ns_core`.
- `notebooks/` — one notebook per experiment (`exp1_convergence`,
  `exp2_conditioning`, `exp3_rank_deficiency`, `exp4_svd_timing`,
  `exp5_time_to_accuracy`, `exp6_cpwl_operator`, `exp7_cpwl_svd_timing`,
  `exp8_cpwl_time_to_accuracy`): edit the config, run, read the plots. Start
  Jupyter from `notebooks/` or the repo root.

  `exp7_cpwl_svd_timing` and `exp8_cpwl_time_to_accuracy` apply exp4's and
  exp5's timing protocols to the CPWL spectral operator of exp6 (any profile
  in `ns_core/cpwl.py`, `"clip"` by default, matching exp6). Every internal
  matrix-sign call inside the operator's sign form is run with
  `sign_map.make_sgn_ns_until` until it reaches a target residual
  `cfg.eps`, instead of exp4's fixed `K_D`; a call that does not reach it
  within `cfg.k_max` is flagged as not converged rather than stopping the
  run. `exp7` additionally prints a results table (one row per degree D):
  relative Frobenius error against the exact CPWL operator (read off a
  signed SVD), evaluation time, the number of internal sign calls actually
  used to build the operator, and the largest iteration count any of those
  calls needed. That call count is empirical, not the thesis's
  `1 + (active positive knots)` lower bound: `cpwl.py`'s sign forms (e.g.
  `clip_map`) call the sign map on every knot shift regardless of its sign,
  so a profile with a non-positive knot (`clip`'s default included) pays one
  more call than the theoretical minimum. `exp8` additionally times the
  exact SVD-based evaluation of the same CPWL operator and can draw it as a
  horizontal reference line (`cfg.svd_reference_line`,
  `plots.add_svd_reference_line`) on the time-vs-degree plot.
- `old_numpy/` — the original NumPy implementation and its numbered
  experiment scripts (figures for the thesis). Standalone; independent of
  `ns_core`. See `old_numpy/README.md`.

## Setup

Python, PyTorch, NumPy, Matplotlib. Install with
`pip install -r requirements.txt`, then run everything from the repo root.
