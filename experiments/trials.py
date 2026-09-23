"""Generic repeated-trial helper: runs a trial function over several
independent, seeded random draws and aggregates mean/std, so every axis
module in this suite reports variability rather than a single run
(reviewer comment: "random experiments should be repeated ... results
should ideally be averaged over several independent matrices, with mean
and variability reported").

Kept separate from ns_core per decision 8 (matrices.py, ns_iteration.py,
etc. operate on a single matrix; repeating/averaging over independent draws
is an experiment-level concern, done by looping, not by batching inside
ns_core) and separate from plotting.py (this aggregates data, it does not
plot it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TypeVar

import torch

T = TypeVar("T")


@dataclass
class TrialSummary:
    """The result of `run_trials`: the per-trial raw values, their mean and
    std, and the seeds used (logged for reproducibility, per the
    reproducibility constraint: every random experiment must log its
    seeds)."""

    values: list[float]
    mean: float
    std: float
    seeds: list[int] = field(default_factory=list)


def run_trials(
    trial_fn: Callable[[torch.Generator], float],
    n_trials: int,
    base_seed: int,
    *,
    device: torch.device | str = "cpu",
) -> TrialSummary:
    """Runs `trial_fn` once per trial, each with an independently seeded
    torch.Generator (seed = base_seed + trial index, so a run is exactly
    reproducible from `base_seed` alone), and aggregates the returned
    scalars.

    `trial_fn` receives the generator and returns a single float (e.g. a
    relative error or a wall-clock time); it is responsible for building
    whatever random matrix/matrices it needs from that generator. `device`
    must match whatever device `trial_fn` draws its random tensors on:
    torch.Generator is itself device-bound (a CPU generator cannot seed a
    torch.randn(..., device="cuda", ...) call), so a trial that draws on
    "cuda" needs device="cuda" here too.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    seeds = [base_seed + i for i in range(n_trials)]
    values = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        values.append(float(trial_fn(generator)))
    t = torch.tensor(values, dtype=torch.float64)
    mean = float(t.mean())
    std = float(t.std(unbiased=False)) if n_trials > 1 else 0.0
    return TrialSummary(values=values, mean=mean, std=std, seeds=seeds)


def run_trials_multi(
    trial_fn: Callable[[torch.Generator], dict[str, float]],
    n_trials: int,
    base_seed: int,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, TrialSummary]:
    """Like `run_trials`, but `trial_fn` returns several named scalars per
    trial (e.g. {"error": ..., "time_s": ...}) from a single random draw,
    so the draw is shared across those quantities instead of redrawn per
    quantity. Returns one TrialSummary per name. See `run_trials` on why
    `device` must match the device `trial_fn` draws its random tensors on.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    seeds = [base_seed + i for i in range(n_trials)]
    per_key: dict[str, list[float]] = {}
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        result = trial_fn(generator)
        for key, val in result.items():
            per_key.setdefault(key, []).append(float(val))
    summaries = {}
    for key, values in per_key.items():
        t = torch.tensor(values, dtype=torch.float64)
        mean = float(t.mean())
        std = float(t.std(unbiased=False)) if n_trials > 1 else 0.0
        summaries[key] = TrialSummary(values=values, mean=mean, std=std, seeds=seeds)
    return summaries
