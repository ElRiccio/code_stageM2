"""Axis 5 of the SVD-comparison suite: dependence on the polynomial degree D
(reviewer: "the influence of the polynomial degree D should also be
studied ... the matrix experiments use D=3 throughout ... compare D=1,2,3,
... numerically").

Sweeps D in {1, 2, 3, 4}, fixed matrix size, reporting accuracy and timing
per map per D. Unlike experiments/rank_deficiency.py's degree sweep (which
reports iterations-to-a-fixed-tolerance for the sign map alone), this one
asks the accuracy/timing question of this suite for every map in the
catalogue, at a fixed NS iteration budget shared across D (N_ITERS_CAP):
higher D is expected to reach a given accuracy in fewer of those iterations
(order D+1 convergence, ns_core.ns_iteration), which is exactly the
trade-off this sweep is meant to make visible.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ns_core import metrics, sign_map
from experiments import map_catalogue, trials
from experiments.svd_accuracy import DEFAULT_BASE_SEED, DEFAULT_M, DEFAULT_N, N_ITERS_CAP, TOL_NS
from experiments.svd_timing import time_call

D_VALUES = [1, 2, 3, 4]
DEFAULT_N_TRIALS = 10


def degree_sweep(
    map_specs: Sequence[map_catalogue.MapSpec],
    D_values: Sequence[int] = D_VALUES,
    m: int = DEFAULT_M,
    n: int = DEFAULT_N,
    *,
    n_iters: int = N_ITERS_CAP,
    tol: float = TOL_NS,
    n_trials: int = DEFAULT_N_TRIALS,
    base_seed: int = DEFAULT_BASE_SEED,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> dict[str, dict[int, dict]]:
    """Accuracy/timing of each map against its exact SVD reference, for each
    D in `D_values`, at fixed matrix size, mean/std over `n_trials`
    independent Gaussian matrices. Returns {map_name: {D: {"error": ...,
    "decomposition_free_s": ...}}} (TrialSummary values).
    """
    results: dict[str, dict[int, dict]] = {spec.name: {} for spec in map_specs}
    for D in D_values:
        for spec in map_specs:

            def trial(generator: torch.Generator, spec=spec, D=D) -> dict[str, float]:
                M = map_catalogue.unit_norm_gaussian(m, n, generator=generator, device=device, dtype=dtype)
                sgn = sign_map.make_sgn_ns(D, n_iters, tol=tol)
                approx = spec.evaluate(M, sgn)
                exact = spec.reference(M)
                error = metrics.relative_frobenius_error(approx, exact).item()
                t_free = time_call(lambda: spec.evaluate(M, sgn), n_repeats=3)
                return {"error": error, "decomposition_free_s": t_free}

            results[spec.name][D] = trials.run_trials_multi(trial, n_trials, base_seed)
    return results
