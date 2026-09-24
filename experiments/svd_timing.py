"""SVD timing experiment: wall-clock time of the decomposition-free msgn
(generalized Newton-Schulz with the bpoly(D) profile, fixed K) against the
SVD-based msgn, over matrix size, sigma_min, device and degree D, each timing
paired with the error against the exact msgn. Every repetition uses a new
random matrix of the same size and sigma_min."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch

from ns_core import metrics, orbit_tools, profiles, sign_map, timing


@dataclass
class SvdTimingConfig:
    """Settings of the SVD timing experiment. Square n x n matrices with
    log-spaced singular values in [smin, 1]. `n_reps` is the number of
    independent random matrices per (n, smin), seeded `seed`, `seed` + 1, ...;
    each route is timed once on each. The `n_warmup` untimed warm-up calls are
    made on the first matrix only. `n_fixed` is the size read by the sigma_min
    plot and must be in `sizes`. `eps` is the accuracy behind the fixed
    iteration count K_D (None: 10 machine epsilons of `dtype`). `power_iters`
    and `power_margin` set the pre-scaling: the scale is `power_margin` times
    the power-iteration estimate of sigma_max, and K_D is computed for
    smin / `power_margin`. `cpu_threads` None keeps the torch default;
    `svd_driver` None keeps torch's choice (CUDA only). Devices must be
    available on the machine, e.g. ["cpu"] without a GPU.

    Usage: cfg = SvdTimingConfig(sizes=[128, 256], devices=["cpu"], n_reps=5)
    """

    sizes: list[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048, 4096])
    smins: list[float] = field(default_factory=lambda: [1e-1, 1e-2, 1e-3, 1e-4])
    degrees: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    devices: list[str] = field(default_factory=lambda: ["cpu", "cuda"])
    n_fixed: int = 1024
    dtype: torch.dtype = torch.float32
    eps: float | None = None
    power_iters: int = 10
    power_margin: float = 1.1
    n_warmup: int = 3
    n_reps: int = 20
    cpu_threads: int | None = None
    svd_driver: str | None = None
    seed: int = 0
    verbose: bool = True


def run_svd_timing(cfg: SvdTimingConfig) -> dict[str, object]:
    """Times the SVD-based and the Newton-Schulz msgn on the same input for
    every (device, size, smin, D), over cfg.n_reps random matrices. Each
    matrix is built once in float64 on the CPU, with its exact msgn N, and
    moved to each device in cfg.dtype. Returns {"cfg": cfg, "eps": eps used,
    "cells": {(device, n, smin): {"svd": {"time": s, "error": e}, "ns": {D:
    {"time": s, "error": e, "K": K}}}}}, where s and e are
    {"median", "mean", "std"} over the matrices: s in seconds, e the relative
    Frobenius error against N (measured outside the timer). K is the number
    of iterations run.

    Usage: res = run_svd_timing(SvdTimingConfig(devices=["cpu"], sizes=[128, 256]))
    """
    eps = 10.0 * torch.finfo(cfg.dtype).eps if cfg.eps is None else cfg.eps
    K = {
        (D, smin): profiles.iteration_count_bound(D, 1.0 - (smin / cfg.power_margin) ** 2, eps)
        for D in cfg.degrees
        for smin in cfg.smins
    }
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
            routes = ["svd", *cfg.degrees]
            samples = {d: {r: {"time": [], "error": []} for r in routes} for d in cfg.devices}
            for rep in range(cfg.n_reps):
                M64, N64 = orbit_tools.make_instance(n, n, None, smin, cfg.seed + rep)
                warm = cfg.n_warmup if rep == 0 else 0
                for device in cfg.devices:
                    M = M64.to(device=device, dtype=cfg.dtype)
                    N = N64.to(device)
                    driver = cfg.svd_driver if torch.device(device).type == "cuda" else None
                    calls = {"svd": lambda: sign_map.sgn_svd(M, driver=driver)}
                    for D in cfg.degrees:
                        calls[D] = lambda D=D: sign_map.sgn_ns_fixed(
                            M, D, K[D, smin], power_iters=cfg.power_iters,
                            margin=cfg.power_margin, generator=generators[device],
                        )
                    for route, fn in calls.items():
                        t, X = timing.time_call(fn, device, warm)
                        samples[device][route]["time"].append(t)
                        samples[device][route]["error"].append(
                            float(metrics.relative_frobenius_error(X.double(), N))
                        )
            for device in cfg.devices:
                s = samples[device]
                cells[device, n, smin] = {
                    "svd": {"time": timing.describe(s["svd"]["time"]), "error": timing.describe(s["svd"]["error"])},
                    "ns": {
                        D: {
                            "time": timing.describe(s[D]["time"]),
                            "error": timing.describe(s[D]["error"]),
                            "K": K[D, smin],
                        }
                        for D in cfg.degrees
                    },
                }
                if cfg.verbose:
                    c = cells[device, n, smin]
                    ns = " | ".join(f"D={D} {1e3 * c['ns'][D]['time']['median']:.2f} ms" for D in cfg.degrees)
                    print(f"  {device}: svd {1e3 * c['svd']['time']['median']:.2f} ms | {ns}")
    finally:
        if cfg.cpu_threads is not None:
            torch.set_num_threads(threads)
    return {"cfg": cfg, "eps": eps, "cells": cells}


def _print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    """Prints a titled text table with aligned columns."""
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    print(title)
    for r in [header, *rows]:
        print("  ".join(x.ljust(w) for x, w in zip(r, widths)))


def time_table(res: dict[str, object], device: str, smin: float) -> None:
    """Prints the time in seconds, "median ± std" over the random matrices,
    for one device and smin: one row per size, one column per D and one for the
    SVD. `res` is the output of `run_svd_timing`.

    Usage: time_table(res, "cpu", 1e-2)
    """
    cfg, cells = res["cfg"], res["cells"]
    fmt = lambda s: f"{s['median']:.3g} ± {s['std']:.2g}"
    rows = [
        [str(n)]
        + [fmt(cells[device, n, smin]["ns"][D]["time"]) for D in cfg.degrees]
        + [fmt(cells[device, n, smin]["svd"]["time"])]
        for n in sorted(cfg.sizes)
    ]
    _print_table(
        f"time (s), median ± std over {cfg.n_reps} matrices: {device}, smin = {smin:g}",
        ["n"] + [f"D={D}" for D in cfg.degrees] + ["SVD"],
        rows,
    )


def accuracy_table(res: dict[str, object], device: str, smin: float) -> None:
    """Prints the accuracy reached for one device and smin: the relative
    Frobenius error against the exact msgn, "mean ± std" over the random
    matrices, for each D and the SVD at every size, and the number of
    iterations K that Newton-Schulz ran (fixed by D and smin). `res` is the
    output of `run_svd_timing`.

    Usage: accuracy_table(res, "cpu", 1e-2)
    """
    cfg, cells = res["cfg"], res["cells"]
    sizes = sorted(cfg.sizes)
    fmt = lambda s: f"{s['mean']:.2e} ± {s['std']:.1e}"
    rows = [
        [f"D={D}", str(cells[device, sizes[0], smin]["ns"][D]["K"])]
        + [fmt(cells[device, n, smin]["ns"][D]["error"]) for n in sizes]
        for D in cfg.degrees
    ]
    rows.append(["SVD", "-"] + [fmt(cells[device, n, smin]["svd"]["error"]) for n in sizes])
    _print_table(
        f"relative Frobenius error, mean ± std over {cfg.n_reps} matrices, and iterations run K: {device}, smin = {smin:g}",
        ["", "K"] + [f"n={n}" for n in sizes],
        rows,
    )
