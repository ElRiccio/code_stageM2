"""Time-to-accuracy for the CPWL operator (exp5's protocol applied to exp6's
operator): how long the SVD-free operator takes until every internal sign call
reaches a target accuracy. Can also time the SVD-based operator as a reference
line.

Mirrors `experiments.time_to_accuracy`, with the CPWL operator in place of the
raw sign map.
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass, field

import torch

from experiments.cpwl_operator import resolve_profile, spectral_reference
from ns_core import matrices, metrics, orbit_tools, sign_map, timing


@dataclass
class CPWLTimeToAccuracyConfig:
    """
    sizes: matrix sizes n (square, full rank)
    conds: condition numbers (largest over smallest singular value) to try
    sigma_max: largest singular value (profile parameters stay in the units of M)
    degrees: degrees to compare
    devices: devices to time (must exist, e.g. ["cpu"] without a GPU)
    dtypes: precisions to try
    profile, alpha, beta, gamma, a, mu, knots, vals: profile choice, as in CPWLOperatorConfig
    eps: target accuracies for every internal sign call
    k_max: step cap per sign call
    svd_reference_line: also time the SVD-based operator
    measure_error: also record the error reached against the exact operator (one extra SVD per matrix)
    power_iters, power_margin, n_warmup, n_reps, cpu_threads, svd_driver, seed, verbose: as in TimeToAccuracyConfig
    Note: a sign call that misses its target is flagged, not fatal
    """

    sizes: list[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048, 4096])
    conds: list[float] = field(default_factory=lambda: [1e1, 1e2, 1e3, 1e4])
    sigma_max: float = 1.0
    degrees: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    devices: list[str] = field(default_factory=lambda: ["cpu", "cuda"])
    dtypes: list[torch.dtype] = field(default_factory=lambda: [torch.float32, torch.float64])
    profile: str = "clip"
    alpha: float | None = -0.5
    beta: float | None = 0.5
    gamma: float | None = 0.3
    a: float | None = 0.2
    mu: float | None = 0.4
    knots: list[float] | None = None
    vals: list[float] | None = None
    eps: list[float] = field(default_factory=lambda: [1e-3, 1e-6])
    k_max: int = 50
    svd_reference_line: bool = True
    measure_error: bool = True
    power_iters: int = 10
    power_margin: float = 1.1
    n_warmup: int = 3
    n_reps: int = 20
    cpu_threads: int | None = None
    svd_driver: str | None = None
    seed: int = 0
    verbose: bool = True


def run_cpwl_time_to_accuracy(cfg: CPWLTimeToAccuracyConfig) -> dict[str, object]:
    """
    cfg: settings
    Returns: {"cfg", "cells": {(device, dtype, n, cond, eps): {D: {"time", "K", "reached", "error"}}},
    "svd": {(device, dtype, n, cond): time stats}}
    Note: time, K and error are {median, mean, std} over matrices; error is relative Frobenius vs
    the exact operator (absent unless measure_error is on); K = largest step count any
    internal call needed; reached = fraction of matrices where every call hit its target;
    "svd" is empty unless svd_reference_line is on
    """
    scalar_fn, sign_form = resolve_profile(cfg)
    generators = {}
    for device in cfg.devices:
        generators[device] = torch.Generator(device=device)
        generators[device].manual_seed(cfg.seed)
    if cfg.cpu_threads is not None:
        threads = torch.get_num_threads()
        torch.set_num_threads(cfg.cpu_threads)
    cells, svd_cells = {}, {}
    try:
        blocks = list(itertools.product(cfg.sizes, cfg.conds))
        for i, (n, cond) in enumerate(blocks, 1):
            if cfg.verbose:
                print(f"[{i}/{len(blocks)}] n={n} cond={cond:g} ({cfg.n_reps} matrices)")
            runs = list(itertools.product(cfg.devices, cfg.dtypes, cfg.eps, cfg.degrees))
            samples = {r: {"time": [], "K": [], "reached": [], "error": []} for r in runs}
            svd_samples = {(device, dtype): [] for device, dtype in itertools.product(cfg.devices, cfg.dtypes)}
            sigma = orbit_tools.log_spectrum(n, cond, cfg.sigma_max)
            for rep in range(cfg.n_reps):
                g = torch.Generator()
                g.manual_seed(cfg.seed + rep)
                M64 = matrices.rand_prescribed_spectrum(n, n, sigma, generator=g, dtype=torch.float64)
                warm = cfg.n_warmup if rep == 0 else 0
                Y64 = spectral_reference(M64.to(cfg.devices[0]), scalar_fn)[-1] if cfg.measure_error else None
                for device, dtype in itertools.product(cfg.devices, cfg.dtypes):
                    M = M64.to(device=device, dtype=dtype)

                    if cfg.svd_reference_line:
                        driver = cfg.svd_driver if torch.device(device).type == "cuda" else None
                        sgn_svd = functools.partial(sign_map.sgn_svd, driver=driver)
                        for _ in range(warm):
                            timing.time_call(lambda: sign_form(M, sgn_svd), device, 0)
                        t, _ = timing.time_call(lambda: sign_form(M, sgn_svd), device, 0)
                        svd_samples[device, dtype].append(t)

                    for eps, D in itertools.product(cfg.eps, cfg.degrees):
                        for _ in range(warm):
                            warm_stats: list[tuple[int, bool]] = []
                            sgn_warm = sign_map.make_sgn_ns_until(
                                D, eps, cfg.k_max, power_iters=cfg.power_iters,
                                margin=cfg.power_margin, generator=generators[device], stats=warm_stats,
                            )
                            timing.time_call(lambda: sign_form(M, sgn_warm), device, 0)
                        stats: list[tuple[int, bool]] = []
                        sgn = sign_map.make_sgn_ns_until(
                            D, eps, cfg.k_max, power_iters=cfg.power_iters,
                            margin=cfg.power_margin, generator=generators[device], stats=stats,
                        )
                        t, Y = timing.time_call(lambda: sign_form(M, sgn), device, 0)
                        s = samples[device, dtype, eps, D]
                        s["time"].append(t)
                        s["K"].append(float(max(k for k, _ in stats)) if stats else 0.0)
                        s["reached"].append(float(all(r for _, r in stats)))
                        if cfg.measure_error:
                            s["error"].append(float(metrics.relative_frobenius_error(Y.double(), Y64.to(device))))
            for device, dtype, eps in itertools.product(cfg.devices, cfg.dtypes, cfg.eps):
                cells[device, dtype, n, cond, eps] = {
                    D: {
                        "time": timing.describe(samples[device, dtype, eps, D]["time"]),
                        "K": timing.describe(samples[device, dtype, eps, D]["K"]),
                        "reached": sum(samples[device, dtype, eps, D]["reached"]) / cfg.n_reps,
                        **(
                            {"error": timing.describe(samples[device, dtype, eps, D]["error"])}
                            if cfg.measure_error
                            else {}
                        ),
                    }
                    for D in cfg.degrees
                }
            if cfg.svd_reference_line:
                for device, dtype in itertools.product(cfg.devices, cfg.dtypes):
                    svd_cells[device, dtype, n, cond] = timing.describe(svd_samples[device, dtype])
    finally:
        if cfg.cpu_threads is not None:
            torch.set_num_threads(threads)
    return {"cfg": cfg, "cells": cells, "svd": svd_cells}
