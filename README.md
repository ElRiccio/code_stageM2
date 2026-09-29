# code_stageM2

Code for my Master's thesis chapter on the matrix sign map, the generalized
Newton-Schulz iteration, and the piecewise-linear spectral operators built on
top of it. It assumes you've read the thesis chapters; none of the theory is
re-derived here.

The main idea being tested: you can compute the sign of a matrix (and from it,
operators like clipping or soft-thresholding the singular values) using only
matrix products, with no SVD. The experiments check how accurate and how fast
that is.

## What's where

- `ns_core/` is the library everything else uses: test matrices, exact
  reference computations, the Newton-Schulz iteration, the sign map, and the
  piecewise-linear (CPWL) operators. Details in `ns_core/README.md`.
- `experiments/` has one module per experiment. Each one has a config
  dataclass and a `run_*` function that returns the results.
- `notebooks/` has one notebook per experiment: change the config, run it,
  look at the plots.
- `old_numpy/` is the original NumPy version, kept only to reproduce the
  thesis figures. It doesn't use `ns_core`. See `old_numpy/README.md`.

## The experiments

| Notebook | Module | What it looks at |
| --- | --- | --- |
| `exp1_convergence` | `convergence.py` | Error per step, for several degrees |
| `exp2_conditioning` | `conditioning.py` | Effect of a small smallest singular value, vs the predicted step count |
| `exp3_rank_deficiency` | `rank_deficiency.py` | Rank-deficient matrices: do the zero singular values stay zero? |
| `exp4_svd_timing` | `svd_timing.py` | Newton-Schulz vs SVD wall-clock time |
| `exp5_time_to_accuracy` | `time_to_accuracy.py` | Time to reach a target accuracy |
| `exp6_cpwl_operator` | `cpwl_operator.py` | The CPWL operator, built on the sign map, vs the exact one |
| `exp7_cpwl_svd_timing` | `cpwl_svd_timing.py` | exp4's timing, applied to the CPWL operator |
| `exp8_cpwl_time_to_accuracy` | `cpwl_time_to_accuracy.py` | exp5's timing, applied to the CPWL operator |

A few things worth knowing:

- **Two ways to stop.** `svd_timing.py` normally runs a fixed number of steps
  (K_D). Set `SvdTimingConfig(use_tolerance=True)` and it stops each matrix
  when its residual reaches `eps` (capped at `k_max`) instead.
- **SVD reference line.** `time_to_accuracy.py` can also time the exact SVD
  sign, if you set `TimeToAccuracyConfig(svd_reference_line=True)`. Draw it
  with `ns_core.plots.add_svd_reference_line`. It's off by default, so no SVD
  is computed unless you ask.
- **exp7 and exp8** run the CPWL operator (any profile from `ns_core/cpwl.py`,
  `"clip"` by default, same as exp6). Every internal sign call runs until it
  reaches `cfg.eps`; one that doesn't get there within `cfg.k_max` is flagged
  as not converged rather than stopping the run. exp8 can also time the exact
  SVD version as a horizontal reference line.
- **exp7's results table** has one row per degree: the error against the
  exact operator, the evaluation time, how many internal sign calls were
  used, and the most steps any single call needed. The call count comes from
  what actually ran, not from the thesis's `1 + (active positive knots)`
  lower bound: the forms in `cpwl.py` (e.g. `clip_map`) call the sign map once
  per knot whatever its sign, so a profile with a non-positive knot (the
  default `clip` has one) costs one extra call.

## Setup

You need Python with PyTorch, NumPy and Matplotlib:

```
pip install -r requirements.txt
```

Run everything from the repo root (except `old_numpy/`, which has its own
instructions). Start Jupyter from the repo root or from `notebooks/`.
