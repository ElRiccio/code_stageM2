"""Time-to-accuracy experiment: how long the Newton-Schulz sign takes to reach a
target accuracy, across size, smallest singular value, device, precision and
degree. Can also time the exact SVD sign as a reference.

Uses the same random matrices as the SVD timing experiment.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch

from experiments.svd_timing import _print_table
from ns_core import matrices, orbit_tools, sign_map, timing


@dataclass
class TimeToAccuracyConfig:
    """
    sizes: matrix sizes n (square, full rank)
    smins: smallest singular values to try
    degrees: degrees to compare
    devices: devices to time (must exist, e.g. ["cpu"] without a GPU)
    dtypes: precisions to try
    eps: target accuracies
    k_max: step cap
    svd_reference_line: also time the exact SVD sign
    power_iters: norm-estimate steps
    power_margin: norm safety factor
    n_warmup: untimed calls, first matrix only
    n_reps: random matrices per (n, smin)
    cpu_threads: torch threads (None = default)
    seed: RNG seed of the first matrix, then +1 each
    verbose: print progress
    Note: a run stops when the residual is below eps (bounds the relative Frobenius
    error); hitting k_max is flagged, and eps below rounding level (~1e-6 in float32)
    will not be reached; timing includes pre-scaling and the residual check
    """

    sizes: list[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048, 4096])
    smins: list[float] = field(default_factory=lambda: [1e-1, 1e-2, 1e-3, 1e-4])
    degrees: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    devices: list[str] = field(default_factory=lambda: ["cpu", "cuda"])
    dtypes: list[torch.dtype] = field(default_factory=lambda: [torch.float32, torch.float64])
    eps: list[float] = field(default_factory=lambda: [1e-3, 1e-6])
    k_max: int = 50
    svd_reference_line: bool = False
    power_iters: int = 10
    power_margin: float = 1.1
    n_warmup: int = 3
    n_reps: int = 20
    cpu_threads: int | None = None
    seed: int = 0
    verbose: bool = True


def run_time_to_accuracy(cfg: TimeToAccuracyConfig) -> dict[str, object]:
    """
    cfg: settings
    Returns: {"cfg", "cells": {(device, dtype, n, smin, eps): {D: {"time", "K", "reached"}}},
    "svd": {(device, dtype, n, smin): time stats}}
    Note: time and K are {median, mean, std} over matrices; reached is the fraction
    that hit eps within k_max; "svd" is empty unless svd_reference_line is on
    """
    generators = {}
    for device in cfg.devices:
        generators[device] = torch.Generator(device=device)
        generators[device].manual_seed(cfg.seed)
    if cfg.cpu_threads is not None:
        threads = torch.get_num_threads()
        torch.set_num_threads(cfg.cpu_threads)
    cells, svd_cells = {}, {}
    try:
        for n, smin in itertools.product(cfg.sizes, cfg.smins):
            if cfg.verbose:
                print(f"n={n} smin={smin:g}: {cfg.n_reps} matrices")
            runs = list(itertools.product(cfg.devices, cfg.dtypes, cfg.eps, cfg.degrees))
            samples = {r: {"time": [], "K": [], "reached": []} for r in runs}
            svd_samples = {(device, dtype): [] for device, dtype in itertools.product(cfg.devices, cfg.dtypes)}
            sigma = orbit_tools.log_spectrum(n, smin)
            for rep in range(cfg.n_reps):
                g = torch.Generator()
                g.manual_seed(cfg.seed + rep)
                M64 = matrices.rand_prescribed_spectrum(n, n, sigma, generator=g, dtype=torch.float64)
                warm = cfg.n_warmup if rep == 0 else 0
                for device, dtype in itertools.product(cfg.devices, cfg.dtypes):
                    M = M64.to(device=device, dtype=dtype)

                    if cfg.svd_reference_line:
                        t, _ = timing.time_call(lambda: sign_map.sgn_svd(M), device, warm)
                        svd_samples[device, dtype].append(t)

                    for eps, D in itertools.product(cfg.eps, cfg.degrees):
                        fn = lambda D=D, eps=eps: sign_map.sgn_ns_until(
                            M, D, eps, cfg.k_max, power_iters=cfg.power_iters,
                            margin=cfg.power_margin, generator=generators[device],
                        )
                        t, (_, K, reached) = timing.time_call(fn, device, warm)
                        s = samples[device, dtype, eps, D]
                        s["time"].append(t)
                        s["K"].append(float(K))
                        s["reached"].append(float(reached))
            for device, dtype, eps in itertools.product(cfg.devices, cfg.dtypes, cfg.eps):
                cells[device, dtype, n, smin, eps] = {
                    D: {
                        "time": timing.describe(samples[device, dtype, eps, D]["time"]),
                        "K": timing.describe(samples[device, dtype, eps, D]["K"]),
                        "reached": sum(samples[device, dtype, eps, D]["reached"]) / cfg.n_reps,
                    }
                    for D in cfg.degrees
                }
                if cfg.verbose:
                    c = cells[device, dtype, n, smin, eps]
                    line = " | ".join(
                        f"D={D} {1e3 * c[D]['time']['median']:.2f} ms (K={c[D]['K']['median']:g})"
                        for D in cfg.degrees
                    )
                    print(f"  {device}, {str(dtype).removeprefix('torch.')}, eps={eps:g}: {line}")
            if cfg.svd_reference_line:
                for device, dtype in itertools.product(cfg.devices, cfg.dtypes):
                    svd_cells[device, dtype, n, smin] = timing.describe(svd_samples[device, dtype])
    finally:
        if cfg.cpu_threads is not None:
            torch.set_num_threads(threads)
    return {"cfg": cfg, "cells": cells, "svd": svd_cells}


def time_to_accuracy_table(
    res: dict[str, object], device: str, dtype: torch.dtype, smin: float, eps: float
) -> None:
    """
    res: output of run_time_to_accuracy
    device, dtype, smin, eps: which setting
    Returns: nothing, prints time (s) as median ± std, one row per size, one column per degree
    Note: * marks a cell where some matrix missed eps
    """
    cfg, cells = res["cfg"], res["cells"]

    def fmt(c):
        return f"{c['time']['median']:.3g} ± {c['time']['std']:.2g}" + ("" if c["reached"] == 1.0 else "*")

    rows = [
        [str(n)] + [fmt(cells[device, dtype, n, smin, eps][D]) for D in cfg.degrees]
        for n in sorted(cfg.sizes)
    ]
    _print_table(
        f"time (s) to reach eps = {eps:g}, median ± std over {cfg.n_reps} matrices: "
        f"{device}, {str(dtype).removeprefix('torch.')}, smin = {smin:g}",
        ["n"] + [f"D={D}" for D in cfg.degrees],
        rows,
    )


def iterations_table(
    res: dict[str, object], device: str, dtype: torch.dtype, smin: float, eps: float
) -> None:
    """
    res: output of run_time_to_accuracy
    device, dtype, smin, eps: which setting
    Returns: nothing, prints steps K as median (mean), one row per size, one column per degree
    """
    cfg, cells = res["cfg"], res["cells"]
    fmt = lambda c: f"{c['K']['median']:g} ({c['K']['mean']:.2f})"
    rows = [
        [str(n)] + [fmt(cells[device, dtype, n, smin, eps][D]) for D in cfg.degrees]
        for n in sorted(cfg.sizes)
    ]
    _print_table(
        f"iterations K to reach eps = {eps:g}, median (mean) over {cfg.n_reps} matrices: "
        f"{device}, {str(dtype).removeprefix('torch.')}, smin = {smin:g}",
        ["n"] + [f"D={D}" for D in cfg.degrees],
        rows,
    )
