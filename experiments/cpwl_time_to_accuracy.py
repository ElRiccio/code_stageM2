"""CPWL time-to-accuracy experiment (exp5, applied to the CPWL profile
studied in exp6): wall-clock time of the decomposition-free CPWL spectral
operator, evaluated through the tolerance-stopped Newton-Schulz sign map,
until every internal sign call reaches a target eps, over matrix size,
sigma_min, device, precision and degree D. An optional horizontal reference
line, the exact SVD-based evaluation time of the same CPWL operator, can be
timed alongside and drawn on the time-vs-degree plot. Mirrors
`experiments.time_to_accuracy`, applied to the CPWL operator of
`experiments.cpwl_operator` instead of the raw sign map."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch

from experiments.cpwl_operator import resolve_profile
from ns_core import matrices, orbit_tools, sign_map, timing


@dataclass
class CPWLTimeToAccuracyConfig:
    """Settings of the CPWL time-to-accuracy experiment. Square n x n
    full-rank matrices with log-spaced singular values in [smin, 1], the
    same sweep as `TimeToAccuracyConfig`. `profile` and its parameters
    (`alpha`, `beta`, `gamma`, `a`, `mu`, `knots`, `vals`) select the CPWL
    map exactly as in `CPWLOperatorConfig`; the defaults reproduce exp6's
    "clip" profile. `eps` lists the tolerances every internal Newton-Schulz
    sign call (`sign_map.sgn_ns_until`) is run to; `k_max` caps its
    iterations, and a call that does not reach its target within `k_max` is
    flagged as not reached rather than stopping the run. `svd_reference_line`
    toggles timing the exact SVD-based CPWL operator once per (device,
    dtype, n, smin), independent of eps and D, for a horizontal line on the
    time-vs-degree plot; when False that timing is skipped. `n_reps`,
    `n_warmup`, `power_iters`, `power_margin` and `cpu_threads` are as in
    `TimeToAccuracyConfig`.
    """

    sizes: list[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048, 4096])
    smins: list[float] = field(default_factory=lambda: [1e-1, 1e-2, 1e-3, 1e-4])
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
    power_iters: int = 10
    power_margin: float = 1.1
    n_warmup: int = 3
    n_reps: int = 20
    cpu_threads: int | None = None
    seed: int = 0
    verbose: bool = True


def run_cpwl_time_to_accuracy(cfg: CPWLTimeToAccuracyConfig) -> dict[str, object]:
    """Times the tolerance-stopped decomposition-free CPWL operator
    (`cfg.profile`'s sign form, evaluated with `sign_map.make_sgn_ns_until`)
    for every (device, dtype, size, smin, eps, D), over `cfg.n_reps` random
    matrices. Each matrix is built once in float64 on the CPU, without any
    decomposition, and moved to each device in each dtype. With
    `cfg.svd_reference_line`, the same operator evaluated exactly through
    `sign_map.sgn_svd` is additionally timed once per (device, dtype, n,
    smin). Returns {"cfg": cfg, "cells": {(device, dtype, n, smin, eps): {D:
    {"time": t, "K": k, "reached": f}}}, "svd": {(device, dtype, n, smin):
    time stats}}, where t and k are {"median", "mean", "std"} over the
    matrices (t in seconds, k the largest iteration count any internal sign
    call needed for that matrix) and f is the fraction of matrices for which
    every internal call reached its target within `cfg.k_max`; "svd" is
    empty unless `cfg.svd_reference_line` is True. Usage:
    run_cpwl_time_to_accuracy(CPWLTimeToAccuracyConfig(sizes=[512], smins=[1e-2])).
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
                        for _ in range(warm):
                            timing.time_call(lambda: sign_form(M, sign_map.sgn_svd), device, 0)
                        t, _ = timing.time_call(lambda: sign_form(M, sign_map.sgn_svd), device, 0)
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
                        t, _ = timing.time_call(lambda: sign_form(M, sgn), device, 0)
                        s = samples[device, dtype, eps, D]
                        s["time"].append(t)
                        s["K"].append(float(max(k for k, _ in stats)) if stats else 0.0)
                        s["reached"].append(float(all(r for _, r in stats)))
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
