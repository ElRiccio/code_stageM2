"""Axis 4 of the SVD-comparison suite: rank / sigma_min dependency at fixed
matrix size (reviewer: "experiments with several ranks and several smallest
nonzero singular values would provide much stronger evidence"; "the
influence of conditioning should be investigated").

This generalizes experiments/rank_deficiency.py's two independent sweeps
(decision 15: rank and conditioning are kept as separate univariate sweeps,
not a combined 2D one) from the matrix sign map alone to all six maps of
this suite's catalogue, but asks a different question: not "how many NS
steps to a fixed tolerance" (that is rank_deficiency.py's question, about
the sign-map iteration itself), but "how accurate/fast is each map's
decomposition-free evaluation against its exact SVD reference" as the
matrix degenerates. Degree is fixed at DEFAULT_D here (not swept): crossing
D with rank/sigma_min would make this a 3-way sweep, out of scope for this
axis (D-dependency is experiments/svd_degree.py's job).
"""

from __future__ import annotations

from typing import Sequence

import torch

from ns_core import matrices, metrics, sign_map
from experiments import map_catalogue, trials
from experiments.svd_accuracy import DEFAULT_BASE_SEED, DEFAULT_D, DEFAULT_M, DEFAULT_N, N_ITERS_CAP, TOL_NS
from experiments.svd_timing import time_call

N_ZERO_VALUES = [0, 6, 12, 18, 23]  # out of r = min(DEFAULT_M, DEFAULT_N) = 48
SIGMA_MIN_VALUES = [0.5, 0.1, 0.01, 1e-3, 1e-4]
DEFAULT_N_TRIALS_SWEEP = 8


def _sweep(
    map_specs: Sequence[map_catalogue.MapSpec],
    swept_values: Sequence,
    build_matrix,
    *,
    m: int,
    n: int,
    D: int,
    n_iters: int,
    tol: float,
    n_trials: int,
    base_seed: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> dict[str, dict]:
    results: dict[str, dict] = {spec.name: {} for spec in map_specs}
    for value in swept_values:
        for spec in map_specs:

            def trial(generator: torch.Generator, spec=spec, value=value) -> dict[str, float]:
                M = build_matrix(value, generator)
                sgn = sign_map.make_sgn_ns(D, n_iters, tol=tol)
                approx = spec.evaluate(M, sgn)
                exact = spec.reference(M)
                error = metrics.relative_frobenius_error(approx, exact).item()
                t_free = time_call(lambda: spec.evaluate(M, sgn), n_repeats=3)
                return {"error": error, "decomposition_free_s": t_free}

            results[spec.name][value] = trials.run_trials_multi(trial, n_trials, base_seed)
    return results


def rank_sweep(
    map_specs: Sequence[map_catalogue.MapSpec],
    n_zero_values: Sequence[int] = N_ZERO_VALUES,
    m: int = DEFAULT_M,
    n: int = DEFAULT_N,
    D: int = DEFAULT_D,
    *,
    n_iters: int = N_ITERS_CAP,
    tol: float = TOL_NS,
    n_trials: int = DEFAULT_N_TRIALS_SWEEP,
    base_seed: int = DEFAULT_BASE_SEED,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> dict[str, dict[int, dict]]:
    """Accuracy/timing vs. rank deficiency (n_zero exact-zero singular
    values, matrices.rand_rank_deficient), one series per map at fixed D."""

    def build_matrix(n_zero: int, generator: torch.Generator) -> torch.Tensor:
        return matrices.rand_rank_deficient(
            m, n, generator=generator, n_zero=n_zero, n_small=0, device=device, dtype=dtype
        )

    return _sweep(
        map_specs, n_zero_values, build_matrix, m=m, n=n, D=D, n_iters=n_iters, tol=tol,
        n_trials=n_trials, base_seed=base_seed, dtype=dtype, device=device,
    )


def conditioning_sweep(
    map_specs: Sequence[map_catalogue.MapSpec],
    sigma_min_values: Sequence[float] = SIGMA_MIN_VALUES,
    m: int = DEFAULT_M,
    n: int = DEFAULT_N,
    D: int = DEFAULT_D,
    *,
    n_iters: int = N_ITERS_CAP,
    tol: float = TOL_NS,
    n_trials: int = DEFAULT_N_TRIALS_SWEEP,
    base_seed: int = DEFAULT_BASE_SEED,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> dict[str, dict[float, dict]]:
    """Accuracy/timing vs. the smallest nonzero singular value sigma_min
    (matrices.rand_rank_deficient, n_small=1), one series per map at fixed
    D."""

    def build_matrix(sigma_min: float, generator: torch.Generator) -> torch.Tensor:
        return matrices.rand_rank_deficient(
            m, n, generator=generator, n_zero=0, n_small=1, s_small=sigma_min, device=device, dtype=dtype
        )

    return _sweep(
        map_specs, sigma_min_values, build_matrix, m=m, n=n, D=D, n_iters=n_iters, tol=tol,
        n_trials=n_trials, base_seed=base_seed, dtype=dtype, device=device,
    )
