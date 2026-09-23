"""Axis 1 of the SVD-comparison suite: accuracy of each map's decomposition-
free evaluation against the exact SVD-based reference (reviewer comment: "a
quantitative table would be useful ... report ||Phihat(M) - Phi(M)||_F /
||Phi(M)||_F").

One relative-Frobenius-error sample per trial, `n_trials` independent
Gaussian matrices (reviewer: "random experiments should be repeated ...
averaged ... with mean and variability reported"), mean/std reported per
map via experiments.trials.run_trials.

The NS surrogate's iteration budget (N_ITERS_CAP, TOL_NS below) is a
generous fixed cap with the same internal tol-based early stop already used
by ns_core.sign_map.make_sgn_ns, matching the convention already established
in experiments/convergence_order.py and experiments/rank_deficiency.py
(rather than a per-matrix theoretical bound from ns_core.profiles: that
bound is stated in terms of the *actual* starting distance sigma_min to the
target, which a plain Gaussian draw does not control, so a bound computed
from an assumed sigma_min would not reliably describe the matrices this
experiment actually draws).
"""

from __future__ import annotations

from typing import Sequence

import torch

from ns_core import metrics, sign_map
from experiments import map_catalogue, trials

DEFAULT_M, DEFAULT_N = 64, 48
DEFAULT_D = 3
N_ITERS_CAP = 40
TOL_NS = 1e-12
DEFAULT_N_TRIALS = 10
DEFAULT_BASE_SEED = 0


def accuracy_experiment(
    map_specs: Sequence[map_catalogue.MapSpec],
    m: int,
    n: int,
    D: int,
    *,
    n_iters: int = N_ITERS_CAP,
    tol: float = TOL_NS,
    n_trials: int = DEFAULT_N_TRIALS,
    base_seed: int = DEFAULT_BASE_SEED,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> dict[str, trials.TrialSummary]:
    """Relative Frobenius error of each map vs. its exact SVD-based
    reference, on independent m x n Gaussian matrices (unit spectral norm),
    at fixed degree D. Returns {map_name: TrialSummary}.
    """
    results: dict[str, trials.TrialSummary] = {}
    for spec in map_specs:

        def trial(generator: torch.Generator, spec=spec) -> float:
            M = map_catalogue.unit_norm_gaussian(m, n, generator=generator, device=device, dtype=dtype)
            sgn = sign_map.make_sgn_ns(D, n_iters, tol=tol)
            approx = spec.evaluate(M, sgn)
            exact = spec.reference(M)
            return metrics.relative_frobenius_error(approx, exact).item()

        results[spec.name] = trials.run_trials(trial, n_trials, base_seed)
    return results
