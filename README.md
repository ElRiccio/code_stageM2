# code_stageM2

Code for the matrix sign map, the generalized Newton-Schulz iteration, and the
piecewise-linear (CPWL) spectral operators built on it. It computes the sign of
a matrix with matrix products only, no SVD.

## Layout

- `ns_core/`: the library (test matrices, references, iteration, sign map,
  CPWL operators). See `ns_core/README.md`.
- `experiments/`: one module per experiment, each with a `Config` dataclass and
  a `run_*` function.
- `notebooks/`: one notebook per experiment. Edit the config, run, read the
  plots.
- `old_numpy/`: the original NumPy version, independent of `ns_core`. See
  `old_numpy/README.md`.

## Experiments

| Notebook | Module | What it does |
| --- | --- | --- |
| `exp1_convergence` | `convergence.py` | Error per step for several degrees |
| `exp2_conditioning` | `conditioning.py` | Effect of the condition number |
| `exp3_rank_deficiency` | `rank_deficiency.py` | Rank-deficient matrices |
| `exp4_svd_timing` | `svd_timing.py` | Newton-Schulz vs SVD time |
| `exp5_time_to_accuracy` | `time_to_accuracy.py` | Time to reach a target accuracy |
| `exp6_cpwl_operator` | `cpwl_operator.py` | CPWL operator vs the exact one |
| `exp7_cpwl_svd_timing` | `cpwl_svd_timing.py` | exp4 for the CPWL operator |
| `exp8_cpwl_time_to_accuracy` | `cpwl_time_to_accuracy.py` | exp5 for the CPWL operator |
| `exp9_polynomial_comparison` | `polynomial_comparison.py` | Muon, Björck and max-derivative quintics, also inside the CPWL operator |
| `exp10_step_cost` | `step_cost.py` | Time of one step against the degree |

Options for each experiment are the fields of its `Config`.

## Setup

```
pip install -r requirements.txt
```

Run everything from the repo root, except `old_numpy/`.
