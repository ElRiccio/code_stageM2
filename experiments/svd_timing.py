"""SVD timing experiment: wall-clock time of the decomposition-free msgn
(generalized Newton-Schulz with the bpoly(D) profile, fixed K) against the
SVD-based msgn, over matrix size, sigma_min, device and degree D, each timing
paired with the error against the exact msgn."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch

from ns_core import metrics, orbit_tools, profiles, sign_map, timing


@dataclass
class SvdTimingConfig:
    """Settings of the SVD timing experiment. Square n x n matrices with
    log-spaced singular values in [smin, 1]. `n_fixed` is the size read by the
    sigma_min plot and must be in `sizes`. `eps` is the accuracy behind the
    fixed iteration count K_D (None: 10 machine epsilons of `dtype`).
    `power_iters` and `power_margin` set the pre-scaling: the scale is
    `power_margin` times the power-iteration estimate of sigma_max, and K_D is
    computed for smin / `power_margin`. `cpu_threads` None keeps the torch
    default; `svd_driver` None keeps torch's choice (CUDA only). Devices must
    be available on the machine, e.g. ["cpu"] without a GPU.

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
    every (device, size, smin, D). The instance of each (size, smin) is built
    once in float64 on the CPU, with its exact msgn N, and moved to each device
    in cfg.dtype. Returns {"cfg": cfg, "eps": eps used, "cells": {(device, n,
    smin): {"svd": {"time": stats, "error": e}, "ns": {D: {"time": stats,
    "error": e, "K": K}}}}}, where stats = {"median", "q25", "q75"} in seconds
    and e = relative Frobenius error against N (measured outside the timer).

    Usage: res = run_svd_timing(SvdTimingConfig(devices=["cpu"], sizes=[128, 256]))
    """
    eps = 10.0 * torch.finfo(cfg.dtype).eps if cfg.eps is None else cfg.eps
    K = {
        (D, smin): profiles.iteration_count_bound(D, 1.0 - (smin / cfg.power_margin) ** 2, eps)
        for D in cfg.degrees
        for smin in cfg.smins
    }
    if cfg.cpu_threads is not None:
        threads = torch.get_num_threads()
        torch.set_num_threads(cfg.cpu_threads)
    cells = {}
    try:
        for n, smin in itertools.product(cfg.sizes, cfg.smins):
            M64, N64 = orbit_tools.make_instance(n, n, None, smin, cfg.seed)
            for device in cfg.devices:
                M = M64.to(device=device, dtype=cfg.dtype)
                N = N64.to(device)
                driver = cfg.svd_driver if torch.device(device).type == "cuda" else None
                stats, X = timing.time_call(
                    lambda: sign_map.sgn_svd(M, driver=driver), device, cfg.n_warmup, cfg.n_reps
                )
                cell = {
                    "svd": {"time": stats, "error": float(metrics.relative_frobenius_error(X.double(), N))},
                    "ns": {},
                }
                g = torch.Generator(device=device)
                g.manual_seed(cfg.seed)
                for D in cfg.degrees:
                    stats, X = timing.time_call(
                        lambda: sign_map.sgn_ns_fixed(
                            M, D, K[D, smin], power_iters=cfg.power_iters, margin=cfg.power_margin, generator=g
                        ),
                        device,
                        cfg.n_warmup,
                        cfg.n_reps,
                    )
                    cell["ns"][D] = {
                        "time": stats,
                        "error": float(metrics.relative_frobenius_error(X.double(), N)),
                        "K": K[D, smin],
                    }
                cells[device, n, smin] = cell
                if cfg.verbose:
                    print(f"{device} n={n} smin={smin:g}: svd {1e3 * cell['svd']['time']['median']:.2f} ms")
    finally:
        if cfg.cpu_threads is not None:
            torch.set_num_threads(threads)
    return {"cfg": cfg, "eps": eps, "cells": cells}


def crossover_table(res: dict[str, object]) -> dict[tuple[str, float], dict[int, int | None]]:
    """Smallest size at which the Newton-Schulz median time is below the SVD
    median time, per (device, smin) and D (None if never). Prints the table
    and returns it as {(device, smin): {D: n or None}}.

    Usage: table = crossover_table(res)
    """
    cfg, cells = res["cfg"], res["cells"]
    table = {}
    for device, smin in itertools.product(cfg.devices, cfg.smins):
        row = {}
        for D in cfg.degrees:
            wins = [
                n
                for n in sorted(cfg.sizes)
                if cells[device, n, smin]["ns"][D]["time"]["median"] < cells[device, n, smin]["svd"]["time"]["median"]
            ]
            row[D] = wins[0] if wins else None
        table[device, smin] = row
    print("smallest n where NS beats the SVD (- = never)")
    print(f"{'device':<8}{'smin':<10}" + "".join(f"D={D:<6}" for D in cfg.degrees))
    for (device, smin), row in table.items():
        print(f"{device:<8}{smin:<10.0e}" + "".join(f"{'-' if row[D] is None else row[D]:<8}" for D in cfg.degrees))
    return table
