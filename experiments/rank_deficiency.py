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
    in [smin, 1]. `tol` is the stopping tolerance on ||X_{k+1} - X_k||_F
    (None: `orbit_tools.default_tol`); once reached the iterate is frozen, so
    rounding noise in the null space is not amplified. `rank_tol` is the
    threshold, relative to the largest singular value, under which a singular
    value of X_k counts as zero (None: `eps`). `degrees` is the only
    list-valued field.

    Usage: cfg = RankDeficiencyConfig(m=128, n=64, rank=16, smin=1e-2)
    """

    m: int = 128
    n: int = 128
    rank: int = 32
    smin: float = 1e-2
    degrees: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    eps: float = 1e-8
    k_max: int = 40
    tol: float | None = None
    rank_tol: float | None = None
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    seed: int = 0


def run_rank_deficiency(cfg: RankDeficiencyConfig) -> dict[str, dict[int, torch.Tensor]]:
    """Runs every degree in cfg.degrees on one rank-r matrix; returns
    {"error": {D: e_0..e_kmax}, "rank": {D: numerical rank of X_0..X_kmax},
    "iterations": {D: first k with error <= eps, or None}}.

    Usage: res = run_rank_deficiency(RankDeficiencyConfig())
    """
    r = orbit_tools.resolve_rank(cfg.m, cfg.n, cfg.rank)
    if r == min(cfg.m, cfg.n):
        raise ValueError("rank must resolve to less than min(m, n)")
    tol = orbit_tools.default_tol(cfg.dtype) if cfg.tol is None else cfg.tol
    rank_tol = cfg.eps if cfg.rank_tol is None else cfg.rank_tol
    M, N = orbit_tools.make_instance(cfg.m, cfg.n, cfg.rank, cfg.smin, cfg.seed, device=cfg.device)
    error, rank, iterations = {}, {}, {}
    for D in cfg.degrees:
        orbit = orbit_tools.run_orbit(M, D, cfg.k_max, cfg.dtype, tol=tol)
        error[D] = orbit_tools.orbit_errors(orbit, N)
        rank[D] = orbit_tools.orbit_ranks(orbit, rank_tol)
        iterations[D] = orbit_tools.first_hit(error[D], cfg.eps)
    return {"error": error, "rank": rank, "iterations": iterations}
