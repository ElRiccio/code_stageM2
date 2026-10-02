"""CPWL operator experiment: how close the SVD-free piecewise-linear spectral
operator gets to the exact one as the iteration count grows, measured both as
an error and as the output spectrum."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from ns_core import cpwl, metrics, orbit_tools, sign_map


@dataclass
class CPWLOperatorConfig:
    """
    m, n: matrix shape
    rank: see orbit_tools.resolve_rank (None = full)
    cond: largest over smallest nonzero singular value
    sigma_max: largest singular value
    spectrum: "log" or "linear" spacing of the singular values
    profile: "clip", "soft", "leaky_relu", "capped_leaky_relu", "leaky_clip" or "spline"
    alpha, beta: clip bounds (also beta = cap for capped_leaky_relu)
    gamma: soft threshold
    a: leaky slope
    mu: leaky clip half-width
    knots, vals: spline points (profile "spline")
    degrees: degrees to compare
    k_max: steps
    eps: relative error at which a run stops (None = orbit_tools.default_eps)
    dtype: precision
    device: cpu or cuda
    seed: RNG seed
    Note: parameters the chosen profile does not use are ignored; the profile parameters
    (alpha, beta, gamma, mu, knots) are in the units of M, not relative to sigma_max
    """

    m: int = 64
    n: int = 48
    rank: int | None = None
    cond: float = 1e2
    sigma_max: float = 1.0
    spectrum: str = "log"
    profile: str = "clip"
    alpha: float | None = -0.5
    beta: float | None = 0.5
    gamma: float | None = 0.3
    a: float | None = 0.2
    mu: float | None = 0.4
    knots: list[float] | None = None
    vals: list[float] | None = None
    degrees: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    k_max: int = 20
    eps: float | None = None
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    seed: int = 0


def resolve_profile(
    cfg: CPWLOperatorConfig,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor, sign_map.msgn], torch.Tensor]]:
    """
    cfg: settings (or any config with the same profile fields)
    Returns: scalar_fn (exact profile), sign_form (M, sgn -> matrix version)
    Note: scalar_fn is the odd extension, because sign forms always map 0 to 0
    """
    if cfg.profile == "clip":
        alpha, beta = cfg.alpha, cfg.beta
        scalar_fn = lambda x: cpwl.clip(x, alpha, beta)
        sign_form = lambda M, sgn: cpwl.clip_map(M, alpha, beta, sgn)
    elif cfg.profile == "soft":
        gamma = cfg.gamma
        scalar_fn = lambda x: cpwl.soft_threshold(x, gamma)
        sign_form = lambda M, sgn: cpwl.soft_map(M, gamma, sgn)
    elif cfg.profile == "leaky_relu":
        a = cfg.a
        scalar_fn = lambda x: cpwl.leaky_relu(x, a)
        sign_form = lambda M, sgn: cpwl.leaky_relu_map(M, a, sgn)
    elif cfg.profile == "capped_leaky_relu":
        a, beta = cfg.a, cfg.beta
        scalar_fn = lambda x: cpwl.capped_leaky_relu(x, a, beta)
        sign_form = lambda M, sgn: cpwl.capped_leaky_relu_map(M, a, beta, sgn)
    elif cfg.profile == "leaky_clip":
        a, mu = cfg.a, cfg.mu
        scalar_fn = lambda x: cpwl.leaky_clip(x, a, mu)
        sign_form = lambda M, sgn: cpwl.leaky_clip_map(M, a, mu, sgn)
    elif cfg.profile == "spline":
        if cfg.knots is None or cfg.vals is None:
            raise ValueError("profile 'spline' requires knots and vals")
        spline = cpwl.PiecewiseLinearProfile(torch.tensor(cfg.knots), torch.tensor(cfg.vals))
        scalar_fn = spline.eval_relu_form
        sign_form = lambda M, sgn: cpwl.spline_map_sign(M, spline, sgn)
    else:
        raise ValueError(f"unknown profile {cfg.profile!r}")
    return cpwl.odd_extension(scalar_fn), sign_form


def spectral_reference(
    M: torch.Tensor, scalar_fn: Callable[[torch.Tensor], torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    M: input matrix
    scalar_fn: exact profile
    Returns: U, sigma, V (SVD of M), target (profile at sigma), Y (exact operator on M)
    Note: near-zero singular values are set to 0, as sgn_svd does
    """
    U, sigma, V = metrics.reference_svd(M)
    tol = metrics.numerical_rank_tol(M, sigma)
    sigma_clean = torch.where(sigma > tol, sigma, torch.zeros_like(sigma))
    target = scalar_fn(sigma_clean)
    Y = (U * target) @ V.mH
    return U, sigma, V, target, Y


def run_cpwl_convergence(cfg: CPWLOperatorConfig) -> dict[str, object]:
    """
    cfg: settings
    Returns: {"error": {D: relative Frobenius error at k = 0..k_max}, "sigma": singular
    values of M, "target": exact profile at sigma, "coords": {D: {k: output spectrum}}}
    Note: every internal sign call uses the same k steps; once the error reaches eps the
    remaining steps are skipped and the error and spectrum are held
    """
    M, _ = orbit_tools.make_instance(
        cfg.m, cfg.n, cfg.rank, cfg.cond, cfg.seed, sigma_max=cfg.sigma_max, spectrum=cfg.spectrum, device=cfg.device
    )
    scalar_fn, sign_form = resolve_profile(cfg)
    U, sigma, V, target, Y_exact = spectral_reference(M, scalar_fn)
    Md = M.to(cfg.dtype)
    eps = orbit_tools.default_eps(cfg.dtype) if cfg.eps is None else cfg.eps

    error, coords = {}, {}
    for D in cfg.degrees:
        e = torch.empty(cfg.k_max + 1, dtype=torch.float64)
        coords[D] = {}
        for k in range(cfg.k_max + 1):
            sgn = sign_map.make_sgn_ns(D, k)
            Y = sign_form(Md, sgn).to(torch.float64)
            e[k] = metrics.relative_frobenius_error(Y, Y_exact)
            coords[D][k] = metrics.spectral_coordinates(Y, U, V)
            if e[k] <= eps:
                e[k + 1 :] = e[k]
                for j in range(k + 1, cfg.k_max + 1):
                    coords[D][j] = coords[D][k]
                break
        error[D] = e
    return {"error": error, "sigma": sigma, "target": target, "coords": coords}
