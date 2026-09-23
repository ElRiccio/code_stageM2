"""Axis 2 of the SVD-comparison suite: wall-clock time of each map's
decomposition-free evaluation vs. its exact SVD-based reference (reviewer:
"there should be a direct comparison ... computation time"; "a quantitative
table ... execution time").

Each trial draws one random matrix and times both routes on it once
`n_repeats` times (after a warm-up call, since the first call on a fresh
matrix can include e.g. BLAS/allocator warm-up not representative of steady
-state cost), taking the minimum of the repeats as that trial's time (the
standard "time a callable" convention: the minimum is the closest single
number to the routine's true cost, since only noise can make a repeat
slower, never faster). `n_trials` independent matrices give the
mean/std reported by experiments.trials.run_trials_multi (reviewer:
repeat and report variability).
"""

from __future__ import annotations

import time
from typing import Sequence

import torch

from ns_core import metrics, sign_map
from experiments import map_catalogue, trials
from experiments.svd_accuracy import (
    DEFAULT_BASE_SEED,
    DEFAULT_D,
    DEFAULT_M,
    DEFAULT_N,
    DEFAULT_N_TRIALS,
    N_ITERS_CAP,
    TOL_NS,
    default_tol_for,
)

N_TIMING_REPEATS = 5


def time_call(fn, n_repeats: int = N_TIMING_REPEATS, *, device: torch.device | str = "cpu") -> float:
    """Minimum wall-clock time, in seconds, of `n_repeats` calls to the
    zero-argument callable `fn`, after one untimed warm-up call.

    CUDA kernel launches are asynchronous: without a synchronize, time.
    perf_counter() around a "cuda" call would measure only launch overhead,
    not the actual device compute time, making GPU calls look implausibly
    fast. When `device` names a CUDA device, this calls
    torch.cuda.synchronize() after the warm-up and after every timed call,
    so the reported time is the real, completed-work time in both cases.
    """
    is_cuda = torch.device(device).type == "cuda"
    fn()  # warm-up, not timed
    if is_cuda:
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        if is_cuda:
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def timing_experiment(
    map_specs: Sequence[map_catalogue.MapSpec],
    m: int,
    n: int,
    D: int,
    *,
    n_iters: int = N_ITERS_CAP,
    tol: float | None = None,
    n_trials: int = DEFAULT_N_TRIALS,
    base_seed: int = DEFAULT_BASE_SEED,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> dict[str, dict[str, trials.TrialSummary]]:
    """Wall-clock time of the decomposition-free evaluation and of the exact
    SVD-based reference, per map, on independent m x n Gaussian matrices at
    fixed degree D.

    Returns {map_name: {"decomposition_free_s": TrialSummary,
    "svd_reference_s": TrialSummary, "speedup": TrialSummary}}, speedup
    being svd_reference_s / decomposition_free_s per trial (so its mean/std
    is a distribution over trials, not the ratio of the two means).

    `tol` defaults to `default_tol_for(dtype)` (see experiments.svd_accuracy)
    rather than the float64-appropriate TOL_NS, so the NS early stop still
    fires at float32 instead of always burning the full `n_iters` cap.
    """
    if tol is None:
        tol = default_tol_for(dtype)
    results: dict[str, dict[str, trials.TrialSummary]] = {}
    for spec in map_specs:

        def trial(generator: torch.Generator, spec=spec) -> dict[str, float]:
            M = map_catalogue.unit_norm_gaussian(m, n, generator=generator, device=device, dtype=dtype)
            sgn = sign_map.make_sgn_ns(D, n_iters, tol=tol)
            t_free = time_call(lambda: spec.evaluate(M, sgn), device=device)
            t_svd = time_call(lambda: spec.reference(M), device=device)
            return {
                "decomposition_free_s": t_free,
                "svd_reference_s": t_svd,
                "speedup": t_svd / t_free if t_free > 0 else float("nan"),
            }

        results[spec.name] = trials.run_trials_multi(trial, n_trials, base_seed, device=device)
    return results
