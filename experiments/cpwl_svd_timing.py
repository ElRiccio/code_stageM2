"""SVD timing for the CPWL operator (exp4's protocol applied to exp6's operator):
how long the SVD-free operator takes compared with evaluating it through an
SVD. Every internal sign call runs to a residual tolerance, not a fixed count.

Same random matrices as `experiments.svd_timing`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch

from experiments.cpwl_operator import resolve_profile, spectral_reference
from experiments.svd_timing import _print_table
from ns_core import metrics, orbit_tools, sign_map, timing


@dataclass
class CPWLSvdTimingConfig:
    """
    sizes: matrix sizes n (square)
    conds: condition numbers (largest over smallest singular value) to try
    sigma_max: largest singular value (profile parameters stay in the units of M)
    degrees: degrees to compare
    devices: devices to time (must exist, e.g. ["cpu"] without a GPU)
    dtype: precision
    profile, alpha, beta, gamma, a, mu, knots, vals: profile choice, as in CPWLOperatorConfig
    eps: residual target for every internal sign call
    k_max: step cap per sign call
    power_iters: norm-estimate steps
    power_margin: norm safety factor
    run_svd: time the SVD-based operator and measure errors against the exact one (False = Newton-Schulz times only, no errors)
    svd_only: time only the SVD-based operator, skip Newton-Schulz
    time_unit: "ms" or "s" in the results table
    time_round_digits: decimals of the table time
    n_warmup: untimed calls, first matrix only
    n_reps: random matrices per (n, cond)
    cpu_threads: torch threads (None = default)
    seed: RNG seed of the first matrix, then +1 each
    verbose: print progress (one line per n, cond)
    Note: a sign call that misses eps is flagged, not fatal
    """

    sizes: list[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048, 4096])
    conds: list[float] = field(default_factory=lambda: [1e1, 1e2, 1e3, 1e4])
    sigma_max: float = 1.0
    degrees: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    devices: list[str] = field(default_factory=lambda: ["cpu", "cuda"])
    dtype: torch.dtype = torch.float32
    profile: str = "clip"
    alpha: float | None = -0.5
    beta: float | None = 0.5
    gamma: float | None = 0.3
    a: float | None = 0.2
    mu: float | None = 0.4
    knots: list[float] | None = None
    vals: list[float] | None = None
    eps: float = 1e-6
    k_max: int = 50
    power_iters: int = 10
    power_margin: float = 1.1
    time_unit: str = "ms"
    time_round_digits: int = 2
    n_warmup: int = 3
    n_reps: int = 20
    cpu_threads: int | None = None
    run_svd: bool = True
    svd_only: bool = False
    seed: int = 0
    verbose: bool = True


def run_cpwl_svd_timing(cfg: CPWLSvdTimingConfig) -> dict[str, object]:
    """
    cfg: settings
    Returns: {"cfg", "cells": {(device, n, cond): {"svd": {"time", "error"},
    "ns": {D: {"time", "error", "calls", "K", "reached"}}}}}
    Note: "svd" is None if run_svd is False, "ns" is empty if svd_only, and errors are then None; time and error are {median, mean, std} over matrices (error is relative
    Frobenius vs the exact float64 operator of the unrounded matrix, so "svd" is the
    dtype SVD-based evaluation's own error); calls = mean internal sign calls, K = largest
    step count any call needed, reached = fraction of matrices where every call hit eps
    """
    if cfg.svd_only and not cfg.run_svd:
        raise ValueError("svd_only needs run_svd")
    degrees = [] if cfg.svd_only else cfg.degrees
    scalar_fn, sign_form = resolve_profile(cfg)
    generators = {}
    for device in cfg.devices:
        generators[device] = torch.Generator(device=device)
        generators[device].manual_seed(cfg.seed)
    if cfg.cpu_threads is not None:
        threads = torch.get_num_threads()
        torch.set_num_threads(cfg.cpu_threads)
    cells = {}
    try:
        blocks = list(itertools.product(cfg.sizes, cfg.conds))
        for i, (n, cond) in enumerate(blocks, 1):
            if cfg.verbose:
                print(f"[{i}/{len(blocks)}] n={n} cond={cond:g} ({cfg.n_reps} matrices)")
            routes = ["svd", *degrees]
            samples = {
                d: {r: {"time": [], "error": [], "calls": [], "K": [], "reached": []} for r in routes}
                for d in cfg.devices
            }
            for rep in range(cfg.n_reps):
                M64, _ = orbit_tools.make_instance(
                    n, n, None, cond, cfg.seed + rep, sigma_max=cfg.sigma_max, with_sign=False
                )
                warm = cfg.n_warmup if rep == 0 else 0
                for device in cfg.devices:
                    M = M64.to(device=device, dtype=cfg.dtype)
                    if cfg.run_svd:
                        Y_exact = spectral_reference(M64.to(device), scalar_fn)[-1]
                        for _ in range(warm):
                            timing.time_call(lambda: sign_form(M, sign_map.sgn_svd), device, 0)
                        t, Y = timing.time_call(lambda: sign_form(M, sign_map.sgn_svd), device, 0)
                        samples[device]["svd"]["time"].append(t)
                        samples[device]["svd"]["error"].append(
                            float(metrics.relative_frobenius_error(Y.double(), Y_exact))
                        )

                    for D in degrees:
                        for _ in range(warm):
                            warm_stats: list[tuple[int, bool]] = []
                            sgn_warm = sign_map.make_sgn_ns_until(
                                D, cfg.eps, cfg.k_max, power_iters=cfg.power_iters,
                                margin=cfg.power_margin, generator=generators[device], stats=warm_stats,
                            )
                            timing.time_call(lambda: sign_form(M, sgn_warm), device, 0)
                        stats: list[tuple[int, bool]] = []
                        sgn = sign_map.make_sgn_ns_until(
                            D, cfg.eps, cfg.k_max, power_iters=cfg.power_iters,
                            margin=cfg.power_margin, generator=generators[device], stats=stats,
                        )
                        t, Y = timing.time_call(lambda: sign_form(M, sgn), device, 0)
                        s = samples[device][D]
                        s["time"].append(t)
                        if cfg.run_svd:
                            s["error"].append(float(metrics.relative_frobenius_error(Y.double(), Y_exact)))
                        s["calls"].append(float(len(stats)))
                        s["K"].append(float(max(k for k, _ in stats)) if stats else 0.0)
                        s["reached"].append(float(all(r for _, r in stats)))
            for device in cfg.devices:
                s = samples[device]
                cells[device, n, cond] = {
                    "svd": (
                        {"time": timing.describe(s["svd"]["time"]), "error": timing.describe(s["svd"]["error"])}
                        if cfg.run_svd
                        else None
                    ),
                    "ns": {
                        D: {
                            "time": timing.describe(s[D]["time"]),
                            "error": timing.describe(s[D]["error"]) if cfg.run_svd else None,
                            "calls": timing.describe(s[D]["calls"])["mean"],
                            "K": int(max(s[D]["K"])) if s[D]["K"] else 0,
                            "reached": sum(s[D]["reached"]) / cfg.n_reps,
                        }
                        for D in degrees
                    },
                }
    finally:
        if cfg.cpu_threads is not None:
            torch.set_num_threads(threads)
    return {"cfg": cfg, "cells": cells}


def cpwl_results_table(res: dict[str, object], device: str, n: int, cond: float) -> None:
    """
    res: output of run_cpwl_svd_timing
    device, n, cond: which setting
    Returns: nothing, prints one row per degree plus SVD: error, time, sign calls, max K
    Note: * on the time marks a cell where some call missed eps; the SVD row is the yardstick for the error
    """
    cfg, cells = res["cfg"], res["cells"]
    cell = cells[device, n, cond]
    scale = 1e3 if cfg.time_unit == "ms" else 1.0
    rows = []
    for D in sorted(cell["ns"]):
        c = cell["ns"][D]
        t = round(scale * c["time"]["median"], cfg.time_round_digits)
        mark = "" if c["reached"] == 1.0 else "*"
        err = f"{c['error']['mean']:.2e}" if c["error"] else "-"
        rows.append([f"D={D}", err, f"{t:g} {cfg.time_unit}{mark}", f"{c['calls']:.1f}", str(c["K"])])
    svd = cell["svd"]
    if svd:
        t = round(scale * svd["time"]["median"], cfg.time_round_digits)
        rows.append(["SVD", f"{svd['error']['mean']:.2e}", f"{t:g} {cfg.time_unit}", "-", "-"])
    _print_table(
        f"CPWL operator ({cfg.profile}), tolerance eps = {cfg.eps:g}, mean/max over {cfg.n_reps} matrices: "
        f"{device}, n = {n}, cond = {cond:g}",
        ["", "rel. Frobenius error", "time", "# sign calls", "max K"],
        rows,
    )
