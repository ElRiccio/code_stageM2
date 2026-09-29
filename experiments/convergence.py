"""Convergence experiment: run the iteration on one matrix for several degrees
and record the error against the exact sign at every step."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ns_core import orbit_tools


@dataclass
class ConvergenceConfig:
    """
    m, n: matrix shape
    rank: see orbit_tools.resolve_rank (None = full)
    smin: smallest nonzero singular value
    degrees: degrees to compare
    k_max: steps
    dtype: precision
    device: cpu or cuda
    seed: RNG seed
    """

    m: int = 128
    n: int = 128
    rank: int | None = None
    smin: float = 1e-2
    degrees: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    k_max: int = 30
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    seed: int = 0


def run_convergence(cfg: ConvergenceConfig) -> dict[str, dict[int, torch.Tensor]]:
    """
    cfg: settings
    Returns: {"error": {D: spectral-norm error at steps 0..k_max}}
    """
    M, N = orbit_tools.make_instance(cfg.m, cfg.n, cfg.rank, cfg.smin, cfg.seed, device=cfg.device)
    error = {}
    for D in cfg.degrees:
        orbit = orbit_tools.run_orbit(M, D, cfg.k_max, cfg.dtype)
        error[D] = orbit_tools.orbit_errors(orbit, N)
    return {"error": error}
