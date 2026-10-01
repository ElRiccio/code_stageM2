"""Time-to-accuracy experiment: how long the Newton-Schulz sign takes to reach a
target accuracy, across size, condition number, device, precision and
degree. Can also time the exact SVD sign as a reference.

Uses the same random matrices as the SVD timing experiment.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch

from experiments.svd_timing import _print_table
from ns_core import matrices, metrics, orbit_tools, sign_map, timing


@dataclass
class TimeToAccuracyConfig:
    """
    sizes: matrix sizes n (square, full rank)
    conds: condition numbers (largest over smallest singular value) to try
    sigma_max: largest singular value
    degrees: degrees to compare
    devices: devices to time (must exist, e.g. ["cpu"] without a GPU)
    dtypes: precisions to try
    eps: target accuracies
    k_max: step cap
    svd_reference_line: also time the exact SVD sign
    measure_error: also record the error reached against the exact sign (one extra SVD per matrix)
    power_iters: norm-estimate steps
    power_margin: norm safety factor
    n_warmup: untimed calls, first matrix only
    n_reps: random matrices per (n, cond)
    cpu_threads: torch threads (None = default)
    svd_driver: CUDA SVD driver (None = torch's choice)
    seed: RNG seed of the first matrix, then +1 each
    verbose: print progress (one line per n, cond)
    Note: a run stops when the residual is below eps (bounds the relative Frobenius
    error); hitting k_max is flagged, and eps below rounding level (~1e-6 in float32)
    will not be reached; timing includes pre-scaling and the residual check
    """

    sizes: list[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048, 4096])
    conds: list[float] = field(default_factory=lambda: [1e1, 1e2, 1e3, 1e4])
    sigma_max: float = 1.0
    degrees: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    devices: list[str] = field(default_factory=lambda: ["cpu", "cuda"])
    dtypes: list[torch.dtype] = field(default_factory=lambda: [torch.float32, torch.float64])
    eps: list[float] = field(default_factory=lambda: [1e-3, 1e-6])
    k_max: int = 50
    svd_reference_line: bool = False
    measure_error: bool = True
    power_iters: int = 10
    power_margin: float = 1.1
    n_warmup: int = 3
    n_reps: int = 20
    cpu_threads: int | None = None
    svd_driver: str | None = None
    seed: int = 0
    verbose: bool = True


def run_time_to_accuracy(cfg: TimeToAccuracyConfig) -> dict[str, object]:
    """
    cfg: settings
    Returns: {"cfg", "cells": {(device, dtype, n, cond, eps): {D: {"time", "K", "reached", "error"}}},
    "svd": {(device, dtype, n, cond): time stats}}
    Note: time, K and error are {median, mean, std} over matrices; error is relative
    Frobenius vs the exact sign, measured outside the timer, and absent unless measure_error
    is on; reached is the fraction that hit eps within k_max; "svd" is empty unless
    svd_reference_line is on
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
                ref_driver = cfg.svd_driver if torch.device(cfg.devices[0]).type == "cuda" else None
                N64 = sign_map.sgn_svd(M64.to(cfg.devices[0]), driver=ref_driver) if cfg.measure_error else None
                for device, dtype in itertools.product(cfg.devices, cfg.dtypes):
                    M = M64.to(device=device, dtype=dtype)
                    driver = cfg.svd_driver if torch.device(device).type == "cuda" else None

                    if cfg.svd_reference_line:
                        t, _ = timing.time_call(lambda: sign_map.sgn_svd(M, driver=driver), device, warm)
                        svd_samples[device, dtype].append(t)

                    for eps, D in itertools.product(cfg.eps, cfg.degrees):
                        fn = lambda D=D, eps=eps: sign_map.sgn_ns_until(
                            M, D, eps, cfg.k_max, power_iters=cfg.power_iters,
                            margin=cfg.power_margin, generator=generators[device],
                        )
                        t, (X, K, reached) = timing.time_call(fn, device, warm)
                        s = samples[device, dtype, eps, D]
                        s["time"].append(t)
                        s["K"].append(float(K))
                        s["reached"].append(float(reached))
                        if cfg.measure_error:
                            s["error"].append(float(metrics.relative_frobenius_error(X.double(), N64.to(device))))
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


def _setting(res: dict[str, object], device: str, dtype: torch.dtype, cond: float) -> str:
    """
    res: output of run_time_to_accuracy (or run_cpwl_time_to_accuracy)
    device, dtype, cond: which setting
    Returns: title tail naming the setting, with the operator first for the CPWL runs
    """
    cfg = res["cfg"]
    tail = f"{device}, {str(dtype).removeprefix('torch.')}, cond = {cond:g}"
    return f"CPWL operator ({cfg.profile}), {tail}" if hasattr(cfg, "profile") else tail


def time_to_accuracy_table(
    res: dict[str, object], device: str, dtype: torch.dtype, cond: float, eps: float
) -> None:
    """
    res: output of run_time_to_accuracy (or run_cpwl_time_to_accuracy)
    device, dtype, cond, eps: which setting
    Returns: nothing, prints time (s) as median ± std, one row per size, one column per degree, plus SVD if it was timed
    Note: * marks a cell where some matrix missed eps
    """
    cfg, cells, svd = res["cfg"], res["cells"], res["svd"]

    def fmt(c):
        return f"{c['time']['median']:.3g} ± {c['time']['std']:.2g}" + ("" if c["reached"] == 1.0 else "*")

    def fmt_svd(n):
        s = svd[device, dtype, n, cond]
        return f"{s['median']:.3g} ± {s['std']:.2g}"

    rows = [
        [str(n)]
        + [fmt(cells[device, dtype, n, cond, eps][D]) for D in cfg.degrees]
        + ([fmt_svd(n)] if svd else [])
        for n in sorted(cfg.sizes)
    ]
    _print_table(
        f"time (s) to reach eps = {eps:g}, median ± std over {cfg.n_reps} matrices: "
        f"{_setting(res, device, dtype, cond)}",
        ["n"] + [f"D={D}" for D in cfg.degrees] + (["SVD"] if svd else []),
        rows,
    )


def speedup_table(res: dict[str, object], device: str, dtype: torch.dtype, cond: float, eps: float) -> None:
    """
    res: output of run_time_to_accuracy (or run_cpwl_time_to_accuracy)
    device, dtype, cond, eps: which setting
    Returns: nothing, prints SVD median time / Newton-Schulz median time, one row per size, one column per degree
    Note: above 1 means Newton-Schulz is faster; needs svd_reference_line
    """
    cfg, cells, svd = res["cfg"], res["cells"], res["svd"]
    if not svd:
        raise ValueError("no SVD times in res: run with svd_reference_line=True")
    rows = [
        [str(n)]
        + [
            f"{svd[device, dtype, n, cond]['median'] / cells[device, dtype, n, cond, eps][D]['time']['median']:.2f}x"
            for D in cfg.degrees
        ]
        for n in sorted(cfg.sizes)
    ]
    _print_table(
        f"speedup over the SVD at eps = {eps:g} (median times): {_setting(res, device, dtype, cond)}",
        ["n"] + [f"D={D}" for D in cfg.degrees],
        rows,
    )


def iterations_table(
    res: dict[str, object], device: str, dtype: torch.dtype, cond: float, eps: float
) -> None:
    """
    res: output of run_time_to_accuracy (or run_cpwl_time_to_accuracy)
    device, dtype, cond, eps: which setting
    Returns: nothing, prints steps K as median (mean), one row per size, one column per degree
    Note: for the CPWL operator K is the largest step count any internal call needed
    """
    cfg, cells = res["cfg"], res["cells"]
    fmt = lambda c: f"{c['K']['median']:g} ({c['K']['mean']:.2f})"
    rows = [
        [str(n)] + [fmt(cells[device, dtype, n, cond, eps][D]) for D in cfg.degrees]
        for n in sorted(cfg.sizes)
    ]
    _print_table(
        f"iterations K to reach eps = {eps:g}, median (mean) over {cfg.n_reps} matrices: "
        f"{_setting(res, device, dtype, cond)}",
        ["n"] + [f"D={D}" for D in cfg.degrees],
        rows,
    )


def time_per_step_table(
    res: dict[str, object], device: str, dtype: torch.dtype, cond: float, eps: float
) -> None:
    """
    res: output of run_time_to_accuracy (or run_cpwl_time_to_accuracy)
    device, dtype, cond, eps: which setting
    Returns: nothing, prints median time / median K in ms, one row per size, one column per degree
    Note: includes the pre-scaling; for the CPWL operator the time covers every internal sign call, so it is not a per-step cost
    """
    cfg, cells = res["cfg"], res["cells"]

    def fmt(c):
        k = c["K"]["median"]
        return f"{1e3 * c['time']['median'] / k:.3g}" if k > 0 else "-"

    rows = [
        [str(n)] + [fmt(cells[device, dtype, n, cond, eps][D]) for D in cfg.degrees]
        for n in sorted(cfg.sizes)
    ]
    _print_table(
        f"time per step (ms) to reach eps = {eps:g}, median time / median K: {_setting(res, device, dtype, cond)}",
        ["n"] + [f"D={D}" for D in cfg.degrees],
        rows,
    )


def error_table(res: dict[str, object], device: str, dtype: torch.dtype, cond: float, eps: float) -> None:
    """
    res: output of run_time_to_accuracy (or run_cpwl_time_to_accuracy)
    device, dtype, cond, eps: which setting
    Returns: nothing, prints the relative Frobenius error reached (mean ± std), one row per size, one column per degree
    Note: needs measure_error; for the CPWL operator the reference is the exact operator
    """
    cfg, cells = res["cfg"], res["cells"]
    if not cfg.measure_error:
        raise ValueError("no errors in res: run with measure_error=True")
    fmt = lambda c: f"{c['error']['mean']:.2e} ± {c['error']['std']:.1e}"
    rows = [
        [str(n)] + [fmt(cells[device, dtype, n, cond, eps][D]) for D in cfg.degrees]
        for n in sorted(cfg.sizes)
    ]
    _print_table(
        f"error reached at eps = {eps:g}, mean ± std over {cfg.n_reps} matrices: {_setting(res, device, dtype, cond)}",
        ["n"] + [f"D={D}" for D in cfg.degrees],
        rows,
    )
