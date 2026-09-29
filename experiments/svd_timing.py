"""SVD timing experiment: how long the Newton-Schulz sign takes compared with
the SVD-based one, across matrix size, smallest singular value, device and
degree. Each timing comes with its error against the exact sign.

Every repetition uses a fresh random matrix of the same size and spectrum.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch

from ns_core import metrics, orbit_tools, profiles, sign_map, timing


@dataclass
class SvdTimingConfig:
    """
    sizes: matrix sizes n (square n x n)
    smins: smallest singular values to try
    degrees: degrees to compare
    devices: devices to time (must exist, e.g. ["cpu"] without a GPU)
    n_fixed: size used by the smin plot (must be in sizes)
    dtype: precision
    eps: accuracy behind K_D (None = 10 machine epsilons)
    use_tolerance: run each matrix until eps instead of a fixed K_D
    k_max: step cap in tolerance mode
    power_iters: norm-estimate steps
    power_margin: norm safety factor
    n_warmup: untimed calls, first matrix only
    n_reps: random matrices per (n, smin)
    cpu_threads: torch threads (None = default)
    svd_driver: CUDA SVD driver (None = torch's choice)
    seed: RNG seed of the first matrix, then +1 each
    verbose: print progress
    Note: in tolerance mode a matrix that misses eps is flagged, not fatal, and K is the median steps run
    """

    sizes: list[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048, 4096])
    smins: list[float] = field(default_factory=lambda: [1e-1, 1e-2, 1e-3, 1e-4])
    degrees: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    devices: list[str] = field(default_factory=lambda: ["cpu", "cuda"])
    n_fixed: int = 1024
    dtype: torch.dtype = torch.float32
    eps: float | None = None
    use_tolerance: bool = False
    k_max: int = 50
    power_iters: int = 10
    power_margin: float = 1.1
    n_warmup: int = 3
    n_reps: int = 20
    cpu_threads: int | None = None
    svd_driver: str | None = None
    seed: int = 0
    verbose: bool = True


def run_svd_timing(cfg: SvdTimingConfig) -> dict[str, object]:
    """
    cfg: settings
    Returns: {"cfg", "eps" (used), "cells": {(device, n, smin): {
    "svd": {"time", "error"}, "ns": {D: {"time", "error", "K"}}}}}
    Note: time and error are {median, mean, std} over matrices; error is relative
    Frobenius vs the exact sign, measured outside the timer; K is the fixed count,
    or in tolerance mode the median steps run plus a "reached" fraction
    """
    eps = 10.0 * torch.finfo(cfg.dtype).eps if cfg.eps is None else cfg.eps
    K = (
        {}
        if cfg.use_tolerance
        else {
            (D, smin): profiles.iteration_count_bound(D, 1.0 - (smin / cfg.power_margin) ** 2, eps)
            for D in cfg.degrees
            for smin in cfg.smins
        }
    )
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
            samples = {
                d: {r: {"time": [], "error": [], "K": [], "reached": []} for r in routes} for d in cfg.devices
            }
            for rep in range(cfg.n_reps):
                M64, N64 = orbit_tools.make_instance(n, n, None, smin, cfg.seed + rep)
                warm = cfg.n_warmup if rep == 0 else 0
                for device in cfg.devices:
                    M = M64.to(device=device, dtype=cfg.dtype)
                    N = N64.to(device)
                    driver = cfg.svd_driver if torch.device(device).type == "cuda" else None
                    for _ in range(warm):
                        timing.time_call(lambda: sign_map.sgn_svd(M, driver=driver), device, 0)
                    t, X = timing.time_call(lambda: sign_map.sgn_svd(M, driver=driver), device, 0)
                    samples[device]["svd"]["time"].append(t)
                    samples[device]["svd"]["error"].append(float(metrics.relative_frobenius_error(X.double(), N)))

                    for D in cfg.degrees:
                        if cfg.use_tolerance:
                            fn = lambda D=D: sign_map.sgn_ns_until(
                                M, D, eps, cfg.k_max, power_iters=cfg.power_iters,
                                margin=cfg.power_margin, generator=generators[device],
                            )
                        else:
                            fn = lambda D=D: sign_map.sgn_ns_fixed(
                                M, D, K[D, smin], power_iters=cfg.power_iters,
                                margin=cfg.power_margin, generator=generators[device],
                            )
                        for _ in range(warm):
                            timing.time_call(fn, device, 0)
                        t, out = timing.time_call(fn, device, 0)
                        X = out[0] if cfg.use_tolerance else out
                        s = samples[device][D]
                        s["time"].append(t)
                        s["error"].append(float(metrics.relative_frobenius_error(X.double(), N)))
                        if cfg.use_tolerance:
                            s["K"].append(float(out[1]))
                            s["reached"].append(float(out[2]))
            for device in cfg.devices:
                s = samples[device]
                cells[device, n, smin] = {
                    "svd": {"time": timing.describe(s["svd"]["time"]), "error": timing.describe(s["svd"]["error"])},
                    "ns": {
                        D: {
                            "time": timing.describe(s[D]["time"]),
                            "error": timing.describe(s[D]["error"]),
                            **(
                                {
                                    "K": int(round(timing.describe(s[D]["K"])["median"])),
                                    "reached": sum(s[D]["reached"]) / cfg.n_reps,
                                }
                                if cfg.use_tolerance
                                else {"K": K[D, smin]}
                            ),
                        }
                        for D in cfg.degrees
                    },
                }
                if cfg.verbose:
                    c = cells[device, n, smin]
                    ns = " | ".join(
                        f"D={D} {1e3 * c['ns'][D]['time']['median']:.2f} ms (K={c['ns'][D]['K']})"
                        for D in cfg.degrees
                    )
                    print(f"  {device}: svd {1e3 * c['svd']['time']['median']:.2f} ms | {ns}")
    finally:
        if cfg.cpu_threads is not None:
            torch.set_num_threads(threads)
    return {"cfg": cfg, "eps": eps, "cells": cells}


def _print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    """
    title: line above the table
    header: column names
    rows: cell strings
    Returns: nothing, prints an aligned table
    """
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    print(title)
    for r in [header, *rows]:
        print("  ".join(x.ljust(w) for x, w in zip(r, widths)))


def time_table(res: dict[str, object], device: str, smin: float) -> None:
    """
    res: output of run_svd_timing
    device: which device
    smin: which smallest singular value
    Returns: nothing, prints time (s) as median ± std, one row per size, one column per degree plus SVD
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
    """
    res: output of run_svd_timing
    device: which device
    smin: which smallest singular value
    Returns: nothing, prints relative Frobenius error (mean ± std) per degree and size, plus the steps K
    Note: in tolerance mode K gets a * if some matrix missed eps
    """
    cfg, cells = res["cfg"], res["cells"]
    sizes = sorted(cfg.sizes)
    fmt = lambda s: f"{s['mean']:.2e} ± {s['std']:.1e}"
    rows = [
        [
            f"D={D}",
            str(cells[device, sizes[0], smin]["ns"][D]["K"])
            + (
                "" if not cfg.use_tolerance or cells[device, sizes[0], smin]["ns"][D]["reached"] == 1.0 else "*"
            ),
        ]
        + [fmt(cells[device, n, smin]["ns"][D]["error"]) for n in sizes]
        for D in cfg.degrees
    ]
    rows.append(["SVD", "-"] + [fmt(cells[device, n, smin]["svd"]["error"]) for n in sizes])
    _print_table(
        f"relative Frobenius error, mean ± std over {cfg.n_reps} matrices, and iterations run K: {device}, smin = {smin:g}",
        ["", "K"] + [f"n={n}" for n in sizes],
        rows,
    )
