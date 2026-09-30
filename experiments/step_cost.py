"""Step cost experiment: how long one Newton-Schulz step takes as a function of
the degree, for the plain step and the Gram-based one. Independent of any
iteration count or accuracy target.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch

from experiments.svd_timing import _print_table
from ns_core import matrices, ns_iteration, timing


@dataclass
class StepCostConfig:
    """
    sizes: matrix sizes n (square n x n)
    degrees: degrees to time
    devices: devices to time (must exist, e.g. ["cpu"] without a GPU)
    dtype: precision
    n_warmup: untimed calls before the timed ones
    n_reps: timed calls per (device, n, D)
    cpu_threads: torch threads (None = default)
    seed: RNG seed of the test matrix
    verbose: print progress (one line per device, n)
    """

    sizes: list[int] = field(default_factory=lambda: [512, 1024, 2048, 4096])
    degrees: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    devices: list[str] = field(default_factory=lambda: ["cpu", "cuda"])
    dtype: torch.dtype = torch.float32
    n_warmup: int = 3
    n_reps: int = 20
    cpu_threads: int | None = None
    seed: int = 0
    verbose: bool = True


def run_step_cost(cfg: StepCostConfig) -> dict[str, object]:
    """
    cfg: settings
    Returns: {"cfg", "cells": {(device, n): {D: {"matrix": stats, "gram": stats}}}}
    Note: stats are {median, mean, std} of seconds over n_reps calls of one step on the same matrix, scaled to norm at most 1
    """
    if cfg.cpu_threads is not None:
        threads = torch.get_num_threads()
        torch.set_num_threads(cfg.cpu_threads)
    cells = {}
    try:
        for device, n in itertools.product(cfg.devices, cfg.sizes):
            if cfg.verbose:
                print(f"{device}: n={n}")
            g = torch.Generator(device=device)
            g.manual_seed(cfg.seed)
            X = matrices.rand_gaussian(n, n, generator=g, device=device, dtype=cfg.dtype)
            X = X / torch.linalg.matrix_norm(X, ord="fro")
            cells[device, n] = {}
            for D in cfg.degrees:
                coeffs = ns_iteration.bpoly_coeffs(D, dtype=cfg.dtype, device=device)
                stats = {}
                for name, step in (("matrix", ns_iteration.ns_step_matrix), ("gram", ns_iteration.ns_step_gram)):
                    for _ in range(cfg.n_warmup):
                        timing.time_call(lambda: step(X, coeffs), device, 0)
                    times = [timing.time_call(lambda: step(X, coeffs), device, 0)[0] for _ in range(cfg.n_reps)]
                    stats[name] = timing.describe(times)
                cells[device, n][D] = stats
    finally:
        if cfg.cpu_threads is not None:
            torch.set_num_threads(threads)
    return {"cfg": cfg, "cells": cells}


def step_cost_table(res: dict[str, object], device: str, variant: str = "gram") -> None:
    """
    res: output of run_step_cost
    device: which device
    variant: "matrix" or "gram"
    Returns: nothing, prints time per step (ms) as median ± std, one row per size, one column per degree
    """
    cfg, cells = res["cfg"], res["cells"]
    fmt = lambda s: f"{1e3 * s['median']:.3g} ± {1e3 * s['std']:.2g}"
    rows = [[str(n)] + [fmt(cells[device, n][D][variant]) for D in cfg.degrees] for n in sorted(cfg.sizes)]
    _print_table(
        f"time per step (ms), median ± std over {cfg.n_reps} calls: {device}, {variant} step",
        ["n"] + [f"D={D}" for D in cfg.degrees],
        rows,
    )
