"""Axis 3 of the SVD-comparison suite: scaling with matrix size (reviewer:
"matrix size should be varied ... a scaling experiment with increasing m, n
would be particularly important").

Sweeps the matrix size at a fixed aspect ratio (m:n = 4:3, matching the
64x48 default used elsewhere in this suite and in the legacy figures),
fixed degree D, and reports both accuracy and timing per size so the same
sweep answers "does the decomposition-free route stay accurate as the
matrix grows" and "where, if anywhere, does it become faster than the SVD"
(reviewer: "the report should ... compare ... dependence on matrix size").

Uses fewer trials at the larger sizes (`n_trials_fn`, default: halve every
doubling, floor 3) since a single SVD at the largest size already costs
much more than the whole small-size sweep; the seed base is still logged
per size so every point stays reproducible.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ns_core import metrics, sign_map
from experiments import map_catalogue, trials
from experiments.svd_accuracy import DEFAULT_BASE_SEED, DEFAULT_D, N_ITERS_CAP, TOL_NS, default_tol_for
from experiments.svd_timing import time_call

SIZES = [(32, 24), (64, 48), (128, 96), (256, 192), (512, 384)]


def _n_trials_for_size(n: int, *, base_trials: int = 10, floor: int = 3) -> int:
    """base_trials halved every doubling past the smallest swept size,
    floored at `floor`: keeps the largest sizes affordable without dropping
    below a handful of independent trials."""
    smallest = SIZES[0][1]
    halvings = 0
    size = smallest
    while size < n:
        size *= 2
        halvings += 1
    return max(floor, base_trials // (2**halvings))


def scaling_experiment(
    map_specs: Sequence[map_catalogue.MapSpec],
    sizes: Sequence[tuple[int, int]] = SIZES,
    D: int = DEFAULT_D,
    *,
    n_iters: int = N_ITERS_CAP,
    tol: float | None = None,
    base_seed: int = DEFAULT_BASE_SEED,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> dict[str, dict[tuple[int, int], dict[str, trials.TrialSummary]]]:
    """For each map and each (m, n) in `sizes`: relative Frobenius error,
    decomposition-free time and exact-SVD time, mean/std over
    `_n_trials_for_size(n)` independent Gaussian matrices.

    Returns {map_name: {(m, n): {"error": ..., "decomposition_free_s": ...,
    "svd_reference_s": ...}}} (TrialSummary values).

    `tol` defaults to `default_tol_for(dtype)` (see experiments.svd_accuracy)
    rather than the float64-appropriate TOL_NS.
    """
    if tol is None:
        tol = default_tol_for(dtype)
    results: dict[str, dict[tuple[int, int], dict[str, trials.TrialSummary]]] = {
        spec.name: {} for spec in map_specs
    }
    for m, n in sizes:
        n_trials = _n_trials_for_size(n)
        for spec in map_specs:

            def trial(generator: torch.Generator, spec=spec, m=m, n=n) -> dict[str, float]:
                M = map_catalogue.unit_norm_gaussian(m, n, generator=generator, device=device, dtype=dtype)
                sgn = sign_map.make_sgn_ns(D, n_iters, tol=tol)
                approx = spec.evaluate(M, sgn)
                exact = spec.reference(M)
                error = metrics.relative_frobenius_error(approx, exact).item()
                t_free = time_call(lambda: spec.evaluate(M, sgn), device=device)
                t_svd = time_call(lambda: spec.reference(M), device=device)
                return {"error": error, "decomposition_free_s": t_free, "svd_reference_s": t_svd}

            results[spec.name][(m, n)] = trials.run_trials_multi(trial, n_trials, base_seed, device=device)
    return results
