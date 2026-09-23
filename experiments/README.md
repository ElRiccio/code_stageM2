# experiments

One module per numerical question raised by the reviewer, built on
`ns_core`. An experiment module exposes a function that runs the
computation and returns its raw results (a dict or small dataclass of
tensors/scalars); it does not plot or save anything. `plotting.py` holds
the shared, experiment-agnostic plotting helpers that turn those results
into figures.

- `plotting.py` — generic plot helpers (spectrum plots, error-decay plots,
  figure saving), each taking plain arrays and labels rather than an
  experiment's own result object.
- `scripts/` — reproducible drivers: call an experiment function, plot its
  results, save the thesis figures. Run as `python -m experiments.scripts.<name>`.
- `notebooks/` — exploratory notebooks, for iterating on an experiment or a
  figure before it becomes a script. Not meant to be reproducible drivers.

## Planned experiment modules

Not yet created; added incrementally as each reviewer-comment area is
tackled. Anticipated, following the protocol above:

- convergence of the generalized Newton–Schulz iteration (order validation,
  rank-deficiency and conditioning sweeps, degree-D comparison)
- evaluation of the CPWL spectral operators (the six-profile catalogue,
  symmetric vs. rectangular forms, stability near knots)
- comparison with SVD-based evaluation (accuracy, iteration/evaluation
  counts, timing, CPU vs. GPU, scaling with matrix size)
