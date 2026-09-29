"""Conditioning experiment: how a small smallest singular value slows the
iteration, and how the observed step count compares with the predicted K_D."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ns_core import orbit_tools, profiles


@dataclass
class ConditioningConfig:
    """
    m, n: matrix shape
    rank: see orbit_tools.resolve_rank (None = full)
    smin: smallest nonzero singular value
    degrees: degrees to compare
    eps: accuracy at which step counts are read
    k_max: steps
    dtype: precision
    device: cpu or cuda
    seed: RNG seed
    """

    m: int = 128
    n: int = 128
    rank: int | None = None
    smin: float = 1e-3
    degrees: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    eps: float = 1e-8
    k_max: int = 40
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    seed: int = 0


def run_conditioning(cfg: ConditioningConfig) -> dict[str, dict[int, object]]:
    """
    cfg: settings
    Returns: {"error": {D: error at steps 0..k_max},
    "predicted": {D: K_D}, "iterations": {D: first step with error <= eps, or None}}
    """
    M, N = orbit_tools.make_instance(cfg.m, cfg.n, cfg.rank, cfg.smin, cfg.seed, device=cfg.device)
    error, predicted, iterations = {}, {}, {}
    for D in cfg.degrees:
        orbit = orbit_tools.run_orbit(M, D, cfg.k_max, cfg.dtype)
        error[D] = orbit_tools.orbit_errors(orbit, N)
        predicted[D] = profiles.iteration_count_bound(D, 1.0 - cfg.smin**2, cfg.eps)
        iterations[D] = orbit_tools.first_hit(error[D], cfg.eps)
    return {"error": error, "predicted": predicted, "iterations": iterations}
