"""Rank-deficiency experiment: run the iteration on a matrix of rank below full,
and check that the error shrinks and the zero singular values stay zero."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ns_core import orbit_tools


@dataclass
class RankDeficiencyConfig:
    """
    m, n: matrix shape
    rank: see orbit_tools.resolve_rank; must be below full (e.g. -1)
    smin: smallest nonzero singular value
    degrees: degrees to compare
    eps: accuracy for the first-hit step and the freeze
    k_max: steps
    rank_tol: relative cutoff for "zero" singular values (None = eps)
    dtype: precision
    device: cpu or cuda
    seed: RNG seed
    Note: the iterate is frozen once the error reaches eps, so rounding noise in the null space is not amplified
    """

    m: int = 128
    n: int = 128
    rank: int = 32
    smin: float = 1e-2
    degrees: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    eps: float = 1e-8
    k_max: int = 40
    rank_tol: float | None = None
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    seed: int = 0


def run_rank_deficiency(cfg: RankDeficiencyConfig) -> dict[str, dict[int, torch.Tensor]]:
    """
    cfg: settings
    Returns: {"error": {D: error per step}, "rank": {D: numerical rank per step},
    "iterations": {D: first step with error <= eps, or None}}
    """
    r = orbit_tools.resolve_rank(cfg.m, cfg.n, cfg.rank)
    if r == min(cfg.m, cfg.n):
        raise ValueError("rank must resolve to less than min(m, n)")
    rank_tol = cfg.eps if cfg.rank_tol is None else cfg.rank_tol
    M, N = orbit_tools.make_instance(cfg.m, cfg.n, cfg.rank, cfg.smin, cfg.seed, device=cfg.device)
    error, rank, iterations = {}, {}, {}
    for D in cfg.degrees:
        orbit, error[D] = orbit_tools.run_orbit(M, N, D, cfg.k_max, cfg.dtype, cfg.eps)
        rank[D] = orbit_tools.orbit_ranks(orbit, rank_tol)
        iterations[D] = orbit_tools.first_hit(error[D], cfg.eps)
    return {"error": error, "rank": rank, "iterations": iterations}
