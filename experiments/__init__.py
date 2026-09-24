"""Convergence experiments for the generalized Newton-Schulz iteration, one
module per experiment, each with a config dataclass and a `run_*` function
that the notebooks in `notebooks/` call.

Modules
-------
convergence     Error decay for several degrees D and its observed rate.
conditioning    Error decay and iteration count against the predicted K_D.
rank_deficiency Rank and zero singular values along the orbit; the limit.
svd_timing      Time of the Newton-Schulz msgn against the SVD-based msgn over
                size, sigma_min, device and D, with the paired error.
"""
