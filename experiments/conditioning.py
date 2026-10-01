"""Conditioning experiment: how a large condition number slows the iteration,
and how the observed step count compares with the predicted K_D. One run covers
several condition numbers."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ns_core import orbit_tools, profiles


@dataclass
class ConditioningConfig:
    """
    m, n: matrix shape
    rank: see orbit_tools.resolve_rank (None = full)
    conds: condition numbers (largest over smallest nonzero singular value) to try
    sigma_max: largest singular value (the iteration runs on M / sigma_max)
    degrees: degrees to compare
    eps: accuracy at which step counts are read and the iterate is frozen (None = 10 machine epsilons)
    k_max: steps
    dtype: precision
    device: cpu or cuda
    seed: RNG seed
    """

    m: int = 128
    n: int = 128
    rank: int | None = None
    conds: list[float] = field(default_factory=lambda: [1e1, 1e2, 1e3, 1e4, 1e5])
    sigma_max: float = 1.0
    degrees: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    eps: float | None = None
    k_max: int = 200
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    seed: int = 0


def run_conditioning(cfg: ConditioningConfig) -> dict[str, object]:
    """
    cfg: settings
    Returns: {"cfg", "eps" (used), "error": {cond: {D: error at steps 0..k_max}},
    "predicted": {cond: {D: K_D}}, "iterations": {cond: {D: first step with error <= eps, or None}}}
    Note: with a tiny eps at a very large cond the freeze may not fire within k_max, which shows as None
    """
    eps = orbit_tools.default_eps(cfg.dtype) if cfg.eps is None else cfg.eps
    error, predicted, iterations = {}, {}, {}
    for cond in cfg.conds:
        M, N = orbit_tools.make_instance(
            cfg.m, cfg.n, cfg.rank, cond, cfg.seed, sigma_max=cfg.sigma_max, normalize=True, device=cfg.device
        )
        error[cond], predicted[cond], iterations[cond] = {}, {}, {}
        for D in cfg.degrees:
            _, error[cond][D] = orbit_tools.run_orbit(M, N, D, cfg.k_max, cfg.dtype, eps)
            predicted[cond][D] = profiles.iteration_count_bound(D, 1.0 - (1.0 / cond) ** 2, eps)
            iterations[cond][D] = orbit_tools.first_hit(error[cond][D], eps)
    return {"cfg": cfg, "eps": eps, "error": error, "predicted": predicted, "iterations": iterations}
