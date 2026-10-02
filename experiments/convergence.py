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
    cond: largest over smallest nonzero singular value
    sigma_max: largest singular value (the iteration runs on M / sigma_max)
    spectrum: "log" or "linear" spacing of the singular values
    degrees: degrees to compare
    k_max: steps
    eps: freeze precision (None = orbit_tools.default_eps)
    dtype: precision
    device: cpu or cuda
    seed: RNG seed
    """

    m: int = 128
    n: int = 128
    rank: int | None = None
    cond: float = 1e2
    sigma_max: float = 1.0
    spectrum: str = "log"
    degrees: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    k_max: int = 30
    eps: float | None = None
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    seed: int = 0


def run_convergence(cfg: ConvergenceConfig) -> dict[str, dict[int, torch.Tensor]]:
    """
    cfg: settings
    Returns: {"error": {D: spectral-norm error at steps 0..k_max}}
    Note: the error is held once it reaches eps
    """
    M, N = orbit_tools.make_instance(
        cfg.m, cfg.n, cfg.rank, cfg.cond, cfg.seed, sigma_max=cfg.sigma_max, spectrum=cfg.spectrum, normalize=True, device=cfg.device
    )
    error = {}
    for D in cfg.degrees:
        _, error[D] = orbit_tools.run_orbit(M, N, D, cfg.k_max, cfg.dtype, cfg.eps)
    return {"error": error}
