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
| `exp2_conditioning` | `conditioning.py` | Effect of the condition number, vs the predicted step count: error curves for one `cond`, and iterations against `cond` over a whole sweep |
| `exp3_rank_deficiency` | `rank_deficiency.py` | Rank-deficient matrices: do the zero singular values stay zero? |
| `exp4_svd_timing` | `svd_timing.py` | Newton-Schulz vs SVD wall-clock time |
| `exp5_time_to_accuracy` | `time_to_accuracy.py` | Time to reach a target accuracy |
| `exp6_cpwl_operator` | `cpwl_operator.py` | The CPWL operator, built on the sign map, vs the exact one |
| `exp7_cpwl_svd_timing` | `cpwl_svd_timing.py` | exp4's timing, applied to the CPWL operator |
| `exp8_cpwl_time_to_accuracy` | `cpwl_time_to_accuracy.py` | exp5's timing, applied to the CPWL operator |
| `exp9_polynomial_comparison` | `polynomial_comparison.py` | Muon, Björck and max-derivative quintics: error per step and the spectrum heading to the sign; also the CPWL operator built on each polynomial (error per step, output spectrum) |
| `exp10_step_cost` | `step_cost.py` | Time of one Newton-Schulz step against the degree (plain and Gram-based step) |

A few things worth knowing:

- **The test spectrum.** Every experiment except exp10 uses singular values
  spaced from `sigma_max` down to `sigma_max / cond`, with random
  singular vectors. The spacing is the `spectrum` field of each Config:
  `"log"` (default) or `"linear"`. `sigma_max` defaults to 1 and `cond` is the condition
  number (the timing and conditioning experiments take a list, `conds`).
  The experiments that run the iteration directly on the matrix (exp1, 2, 3,
  9's polynomial part) divide by `sigma_max` first, so changing it does not
  change their results. The timing experiments rescale through the norm
  estimate (`power_margin`), so they are unaffected too. The CPWL experiments
  (exp6, 7, 8, 9's CPWL part) are the exception: the profile parameters
  (`alpha`, `beta`, `gamma`, `mu`, `knots`) are in the units of the matrix,
  so `sigma_max` changes where the profile cuts the spectrum.

- **Two ways to stop.** The experiments that compare against the exact sign
  (exp1, exp2, exp3, exp6, exp9) know the error at every step, so once it reaches
  `eps` the iterate is frozen and its error is held until `k_max`. The timing
  experiments (exp4, exp5, exp7, exp8) can't afford that reference, so they stop
  on a residual instead, `||X (X^T X - I)||_F <= eps sqrt(r)`. It is zero at
  singular values 0 and 1, so it also works on rank-deficient matrices
  (singular values below about `eps` count as zero, as in `sgn_svd`).
  `svd_timing.py` normally runs a fixed number of steps (K_D); set
  `SvdTimingConfig(use_tolerance=True)` to use the residual (capped at `k_max`).
- **Why stop at all.** In floating point the zero singular values of a
  rank-deficient matrix are not exactly zero, and the iteration multiplies
  them by about 3/2 (degree 1) per step until they are promoted to 1. That is
  after roughly 40 steps in float32 and 90 in float64. A fixed step count or
  `k_max` therefore has to be long enough for the smallest genuine singular
  value and short enough not to reach that point. The noise also sets the
  accuracy floor: in float32 a rank-deficient run bottoms out near 1e-5, so
  `eps` there should be around 1e-4, not the default of 10 machine epsilons.
- **SVD reference line.** `time_to_accuracy.py` can also time the exact SVD
  sign, if you set `TimeToAccuracyConfig(svd_reference_line=True)` (same for
  exp8). `ns_core.plots.plot_time_vs_degree` then draws it as a dashed line by
  itself; pass `show_svd=False` to hide it. It's off by default, so no SVD
  is computed unless you ask.
- **exp7 and exp8** run the CPWL operator (any profile from `ns_core/cpwl.py`,
  `"clip"` by default, same as exp6). Every internal sign call runs until it
  reaches `cfg.eps`; one that doesn't get there within `cfg.k_max` is flagged
  as not converged rather than stopping the run. exp8 can also time the exact
  SVD version as a horizontal reference line.
- **Optional SVD in exp4 and exp7.** `run_svd=False` drops every SVD call:
  no SVD timing and no exact reference, so Newton-Schulz is timed but its
  errors are `None` (tables show `-`, the accuracy and speedup tables are
  skipped). `svd_only=True` does the opposite and times only the SVD (needs
  `run_svd`), so tables and plots show just the SVD column/curve.
- **Progress vs tables.** With `verbose=True` the timing experiments only
  print one progress line per (n, cond); all results are in tables.
  - exp4: time, speedup over the SVD, iterations K, error, and time against
    cond at a chosen size.
  - exp5: time (with an SVD column if `svd_reference_line`), speedup over the
    SVD, iterations, time per step, and the error actually reached against the
    exact sign (mean ± std; `measure_error`, costs one extra SVD per matrix).
  - exp7: exp4's time, speedup and cond tables, plus a results table per cond
    with one row per degree and an SVD row (the SVD-based evaluation in the
    same dtype, the yardstick for the error; ~1e-16 in float64). Errors are
    against the exact float64 operator of the unrounded matrix. Columns: the
    error, the evaluation time, how many internal sign calls were used, and
    the most steps any single call needed.
  - exp8: exp5's tables, read off the CPWL results (they share a layout); the
    error is against the exact operator. Its time per step covers all internal
    calls, so it is not a pure step cost.
- **exp7's call count** comes from what actually ran, not from the thesis's
  `1 + (active positive knots)` lower bound: the forms in `cpwl.py` (e.g.
  `clip_map`) call the sign map once per knot whatever its sign, so a profile
  with a non-positive knot (the default `clip` has one) costs one extra call.

## Setup

You need Python with PyTorch, NumPy and Matplotlib:

```
pip install -r requirements.txt
```

Run everything from the repo root (except `old_numpy/`, which has its own
instructions). Start Jupyter from the repo root or from `notebooks/`.
