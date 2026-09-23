"""Axis 6 of the SVD-comparison suite: CPU vs. GPU execution (reviewer: "the
report motivates the approach partly by the efficiency of matrix
multiplication on GPUs, but no GPU experiment is presented ... some
numerical evidence would be highly desirable").

Runs the same accuracy/timing comparison as experiments.svd_timing on both
"cpu" and, if torch.cuda.is_available(), "cuda", at a larger default matrix
size than the rest of this suite (DEFAULT_SIZE below): a GPU's throughput
advantage over CPU LAPACK/BLAS is a large-matrix effect, invisible (or
reversed, by launch overhead) at the small sizes used for the
accuracy/timing baselines elsewhere in this suite. float32 is used here
(rather than this suite's usual float64) since GPU throughput, not
numerical accuracy, is what this axis is measuring, and float32 is the
dtype most GPU hardware is fastest at.

If no CUDA device is available (true in the sandbox this suite was
developed and run in), `device_experiment` runs the CPU side only and
reports `cuda_available=False`; callers should treat the "cuda" column as
missing rather than as a same-hardware comparison in that case.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ns_core import metrics, sign_map
from experiments import map_catalogue, trials
from experiments.svd_accuracy import DEFAULT_BASE_SEED, DEFAULT_D
from experiments.svd_timing import time_call

DEFAULT_SIZE = (512, 384)
N_ITERS_CAP = 40
TOL_NS = 1e-6  # float32: no point chasing a float64-scale tolerance
DEFAULT_N_TRIALS = 5


def device_experiment(
    map_specs: Sequence[map_catalogue.MapSpec],
    m: int = DEFAULT_SIZE[0],
    n: int = DEFAULT_SIZE[1],
    D: int = DEFAULT_D,
    *,
    devices: Sequence[str] = ("cpu", "cuda"),
    n_iters: int = N_ITERS_CAP,
    tol: float = TOL_NS,
    n_trials: int = DEFAULT_N_TRIALS,
    base_seed: int = DEFAULT_BASE_SEED,
    dtype: torch.dtype = torch.float32,
) -> dict[str, dict[str, dict]]:
    """Accuracy/timing of each map, per available device in `devices`
    (entries other than "cpu" are skipped when torch.cuda.is_available() is
    False). Returns {map_name: {device: {"error": ..., "time_s": ...}}}
    (TrialSummary values); a skipped device is simply absent from the
    result rather than reported as a failure.
    """
    available = [d for d in devices if d == "cpu" or torch.cuda.is_available()]
    results: dict[str, dict[str, dict]] = {spec.name: {} for spec in map_specs}
    for device in available:
        for spec in map_specs:

            def trial(generator: torch.Generator, spec=spec, device=device) -> dict[str, float]:
                M = map_catalogue.unit_norm_gaussian(m, n, generator=generator, device=device, dtype=dtype)
                sgn = sign_map.make_sgn_ns(D, n_iters, tol=tol)
                approx = spec.evaluate(M, sgn)
                exact = spec.reference(M)
                error = metrics.relative_frobenius_error(approx, exact).item()

                def _sync_call(fn):
                    out = fn()
                    if device == "cuda":
                        torch.cuda.synchronize()
                    return out

                t_free = time_call(lambda: _sync_call(lambda: spec.evaluate(M, sgn)), n_repeats=3)
                return {"error": error, "time_s": t_free}

            results[spec.name][device] = trials.run_trials_multi(trial, n_trials, base_seed)
    return results
