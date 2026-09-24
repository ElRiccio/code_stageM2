"""Convergence experiment: one matrix, the error of X_{k+1} = I_D(X_k) against
msgn(M) for several degrees D, to read the observed order and constant."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ns_core import orbit_tools


@dataclass
class ConvergenceConfig:
    """Settings of the convergence experiment. `rank` follows
    `orbit_tools.resolve_rank` (None = full rank); the nonzero singular values
    are log-spaced in [smin, 1]. `degrees` is the only list-valued field.

    Usage: cfg = ConvergenceConfig(m=128, n=64, smin=1e-2, degrees=[1, 2, 3])
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
    """Runs every degree in cfg.degrees on one matrix; returns
    {"error": {D: e_0..e_kmax}}, the spectral-norm error against msgn(M).

    Usage: res = run_convergence(ConvergenceConfig())
    """
    M, N = orbit_tools.make_instance(cfg.m, cfg.n, cfg.rank, cfg.smin, cfg.seed, device=cfg.device)
    error = {}
    for D in cfg.degrees:
        orbit = orbit_tools.run_orbit(M, D, cfg.k_max, cfg.dtype)
        error[D] = orbit_tools.orbit_errors(orbit, N)
    return {"error": error}
