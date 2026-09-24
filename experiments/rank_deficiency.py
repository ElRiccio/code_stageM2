"""Rank-deficiency experiment: one matrix of rank r < min(m, n); the error
against msgn(M), and whether the rank and the zero singular values are kept
along the orbit."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ns_core import orbit_tools


@dataclass
class RankDeficiencyConfig:
    """Settings of the rank-deficiency experiment. `rank` follows
    `orbit_tools.resolve_rank` and must resolve to less than min(m, n)
    (e.g. -1 for min(m, n) - 1); the r nonzero singular values are log-spaced
    in [smin, 1]. `degrees` is the only list-valued field.

    Usage: cfg = RankDeficiencyConfig(m=128, n=64, rank=16, smin=1e-2)
    """

    m: int = 128
    n: int = 128
    rank: int = 32
    smin: float = 1e-2
    degrees: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    eps: float = 1e-8
    k_max: int = 40
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    seed: int = 0


def run_rank_deficiency(cfg: RankDeficiencyConfig) -> dict[str, dict[int, torch.Tensor]]:
    """Runs every degree in cfg.degrees on one rank-r matrix; returns
    {"error": ..., "trailing": ..., "smallest": ...}, each {D: value at
    k = 0..k_max}: the error against msgn(M), the largest singular value
    beyond the r-th, and the r-th singular value of X_k.

    Usage: res = run_rank_deficiency(RankDeficiencyConfig())
    """
    r = orbit_tools.resolve_rank(cfg.m, cfg.n, cfg.rank)
    if r == min(cfg.m, cfg.n):
        raise ValueError("rank must resolve to less than min(m, n)")
    M, N = orbit_tools.make_instance(cfg.m, cfg.n, cfg.rank, cfg.smin, cfg.seed, device=cfg.device)
    error, trailing, smallest = {}, {}, {}
    for D in cfg.degrees:
        orbit = orbit_tools.run_orbit(M, D, cfg.k_max, cfg.dtype)
        sv = orbit_tools.orbit_singular_values(orbit)
        error[D] = orbit_tools.orbit_errors(orbit, N)
        trailing[D] = sv[:, r:].amax(dim=1)
        smallest[D] = sv[:, r - 1]
    return {"error": error, "trailing": trailing, "smallest": smallest}
