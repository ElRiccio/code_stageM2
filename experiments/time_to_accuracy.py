"""Time-to-accuracy experiment: wall-clock time of the decomposition-free msgn
(generalized Newton-Schulz with the bpoly(D) profile) until an SVD-free
residual reaches a target accuracy eps, over matrix size, sigma_min, device,
precision and degree D. Every repetition uses a new random matrix of the same
size and sigma_min, the same matrices as the SVD timing experiment."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch

from experiments.svd_timing import _print_table
from ns_core import matrices, orbit_tools, sign_map, timing


@dataclass
class TimeToAccuracyConfig:
    """Settings of the time-to-accuracy experiment. Square n x n full-rank
    matrices with log-spaced singular values in [smin, 1]. `n_reps` is the
    number of independent random matrices per (n, smin), seeded `seed`,
    `seed` + 1, ...; each (device, dtype, eps, D) is timed once on each. The
    `n_warmup` untimed warm-up calls are made on the first matrix only. `eps`
    lists the target accuracies: a run stops once ||I - X^H X||_F / sqrt(n) <=
    eps, an upper bound on the relative Frobenius error against the exact
    msgn. `k_max` caps the iterations; a run that hits it is flagged as not
    reached, so targets below the rounding level of a dtype (about 1e-6 in
    float32) should not be requested for it. `power_iters` and `power_margin`
    set the pre-scaling: the scale is `power_margin` times the power-iteration
    estimate of sigma_max. `cpu_threads` None keeps the torch default. Devices
    must be available on the machine, e.g. ["cpu"] without a GPU. The timed
    region holds the pre-scaling and the iteration with its residual check.

    Usage: cfg = TimeToAccuracyConfig(sizes=[128, 256], devices=["cpu"], eps=[1e-6], n_reps=5)
    """

    sizes: list[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048, 4096])
    smins: list[float] = field(default_factory=lambda: [1e-1, 1e-2, 1e-3, 1e-4])
    degrees: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    devices: list[str] = field(default_factory=lambda: ["cpu", "cuda"])
    dtypes: list[torch.dtype] = field(default_factory=lambda: [torch.float32, torch.float64])
    eps: list[float] = field(default_factory=lambda: [1e-3, 1e-6])
    k_max: int = 50
    power_iters: int = 10
    power_margin: float = 1.1
    n_warmup: int = 3
    n_reps: int = 20
    cpu_threads: int | None = None
    seed: int = 0
    verbose: bool = True


def run_time_to_accuracy(cfg: TimeToAccuracyConfig) -> dict[str, object]:
    """Times the Newton-Schulz msgn stopped at each target in cfg.eps, for
    every (device, dtype, size, smin, eps, D), over cfg.n_reps random
    matrices. Each matrix is built once in float64 on the CPU, without any
    decomposition, and moved to each device in each dtype. Returns {"cfg":
    cfg, "cells": {(device, dtype, n, smin, eps): {D: {"time": t, "K": k,
    "reached": f}}}}, where t and k are {"median", "mean", "std"} over the
    matrices (t in seconds, k the number of iterations run) and f is the
    fraction of matrices that reached eps within cfg.k_max.

    Usage: res = run_time_to_accuracy(TimeToAccuracyConfig(devices=["cpu"], sizes=[128, 256]))
    """
    generators = {}
    for device in cfg.devices:
        generators[device] = torch.Generator(device=device)
        generators[device].manual_seed(cfg.seed)
    if cfg.cpu_threads is not None:
        threads = torch.get_num_threads()
        torch.set_num_threads(cfg.cpu_threads)
    cells = {}
    try:
        for n, smin in itertools.product(cfg.sizes, cfg.smins):
            if cfg.verbose:
                print(f"n={n} smin={smin:g}: {cfg.n_reps} matrices")
            runs = list(itertools.product(cfg.devices, cfg.dtypes, cfg.eps, cfg.degrees))
            samples = {r: {"time": [], "K": [], "reached": []} for r in runs}
            sigma = orbit_tools.log_spectrum(n, smin)
            for rep in range(cfg.n_reps):
                g = torch.Generator()
                g.manual_seed(cfg.seed + rep)
                M64 = matrices.rand_prescribed_spectrum(n, n, sigma, generator=g, dtype=torch.float64)
                warm = cfg.n_warmup if rep == 0 else 0
                for device, dtype in itertools.product(cfg.devices, cfg.dtypes):
                    M = M64.to(device=device, dtype=dtype)
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
    finally:
        if cfg.cpu_threads is not None:
            torch.set_num_threads(threads)
    return {"cfg": cfg, "cells": cells}


def time_to_accuracy_table(
    res: dict[str, object], device: str, dtype: torch.dtype, smin: float, eps: float
) -> None:
    """Prints the time in seconds to reach eps, "median ± std" over the random
    matrices, for one device, dtype and smin: one row per size, one column per
    D. A trailing * marks a cell where some matrix did not reach eps within
    cfg.k_max. `res` is the output of `run_time_to_accuracy`.

    Usage: time_to_accuracy_table(res, "cpu", torch.float64, 1e-2, 1e-9)
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
    """Prints the number of iterations K run to reach eps, "median (mean)"
    over the random matrices, for one device, dtype and smin: one row per size,
    one column per D. `res` is the output of `run_time_to_accuracy`.

    Usage: iterations_table(res, "cpu", torch.float64, 1e-2, 1e-9)
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
