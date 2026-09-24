"""Conditioning experiment: one matrix with a given smin, the error decay for
several degrees D and the iteration count against the predicted K_D."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ns_core import orbit_tools, profiles


@dataclass
class ConditioningConfig:
    """Settings of the conditioning experiment. `eps` is the accuracy at which
    the observed and predicted iteration counts are read; `degrees` is the
    only list-valued field.

    Usage: cfg = ConditioningConfig(smin=1e-3, eps=1e-8)
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
    """Runs every degree in cfg.degrees on one matrix; returns
    {"error": {D: e_0..e_kmax}, "predicted": {D: K_D(smin, eps)},
    "iterations": {D: first k with error <= eps, or None}}.

    Usage: res = run_conditioning(ConditioningConfig())
    """
    M, N = orbit_tools.make_instance(cfg.m, cfg.n, cfg.rank, cfg.smin, cfg.seed, device=cfg.device)
    error, predicted, iterations = {}, {}, {}
    for D in cfg.degrees:
        orbit = orbit_tools.run_orbit(M, D, cfg.k_max, cfg.dtype)
        error[D] = orbit_tools.orbit_errors(orbit, N)
        predicted[D] = profiles.iteration_count_bound(D, 1.0 - cfg.smin**2, cfg.eps)
        iterations[D] = orbit_tools.first_hit(error[D], cfg.eps)
    return {"error": error, "predicted": predicted, "iterations": iterations}
