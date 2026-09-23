"""experiments: one module per numerical question, built on ns_core.

Each experiment is a function that runs a computation and returns its raw
results (tensors and scalars, usually as a dict or a small dataclass) — it
does not save figures itself. Plotting lives separately in
experiments.plotting, so the same results can feed more than one figure, a
table, or be replotted without rerunning the computation.

experiments/scripts/   reproducible drivers that call an experiment
                        function, plot its results, and save the thesis
                        figures to disk.
experiments/notebooks/ exploratory notebooks, not meant to be reproducible
                        drivers.

Experiment modules are added incrementally, one reviewer-comment question at
a time; see the project README for the current list.
"""
