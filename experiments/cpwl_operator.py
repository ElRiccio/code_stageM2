"""CPWL operator experiment: convergence of the decomposition-free entrywise
CPWL spectral operator, evaluated through the iterative msgn surrogate, to
the exact operator read off a signed SVD, and the convergence of its output
spectrum as the iteration count grows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from ns_core import cpwl, metrics, orbit_tools, sign_map


@dataclass
class CPWLOperatorConfig:
    """Settings of the CPWL operator experiment. `rank` follows
    `orbit_tools.resolve_rank` (None = full rank); the nonzero singular
    values of the test matrix are log-spaced in [smin, 1].

    `profile` selects the CPWL map and reads its parameters from the fields
    below: "clip" (alpha, beta), "soft" (gamma), "leaky_relu" (a),
    "capped_leaky_relu" (a, beta), "leaky_clip" (a, mu), or "spline" (knots,
    vals) for an arbitrary `cpwl.PiecewiseLinearProfile`. Unused fields for
    the chosen profile are ignored.

    `degrees` and `k_max` drive the error-against-iterations run (one curve
    per degree, k = 0..k_max); `D_show` and `k_show` drive the spectrum run
    (one degree, a few iteration counts). `tol` is the stopping tolerance on
    every internal Newton-Schulz sign call (None: `orbit_tools.default_tol`).
    """

    m: int = 64
    n: int = 48
    rank: int | None = None
    smin: float = 1e-2
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
    D_show: int = 3
    k_show: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    tol: float | None = None
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    seed: int = 0


def resolve_profile(
    cfg: CPWLOperatorConfig,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor, sign_map.msgn], torch.Tensor]]:
    """The exact scalar profile and its decomposition-free sign-form lift
    selected by cfg.profile, bound to cfg's parameters: (scalar_fn,
    sign_form), with scalar_fn(sigma) the exact map on singular values and
    sign_form(M, sgn) the matrix expression built from a sign callable.
    Usage: scalar_fn, sign_form = resolve_profile(cfg); sign_form(M, sgn).
    """
    if cfg.profile == "clip":
        alpha, beta = cfg.alpha, cfg.beta
        return (
            lambda x: cpwl.clip(x, alpha, beta),
            lambda M, sgn: cpwl.clip_map(M, alpha, beta, sgn),
        )
    if cfg.profile == "soft":
        gamma = cfg.gamma
        return (
            lambda x: cpwl.soft_threshold(x, gamma),
            lambda M, sgn: cpwl.soft_map(M, gamma, sgn),
        )
    if cfg.profile == "leaky_relu":
        a = cfg.a
        return (
            lambda x: cpwl.leaky_relu(x, a),
            lambda M, sgn: cpwl.leaky_relu_map(M, a, sgn),
        )
    if cfg.profile == "capped_leaky_relu":
        a, beta = cfg.a, cfg.beta
        return (
            lambda x: cpwl.capped_leaky_relu(x, a, beta),
            lambda M, sgn: cpwl.capped_leaky_relu_map(M, a, beta, sgn),
        )
    if cfg.profile == "leaky_clip":
        a, mu = cfg.a, cfg.mu
        return (
            lambda x: cpwl.leaky_clip(x, a, mu),
            lambda M, sgn: cpwl.leaky_clip_map(M, a, mu, sgn),
        )
    if cfg.profile == "spline":
        if cfg.knots is None or cfg.vals is None:
            raise ValueError("profile 'spline' requires knots and vals")
        spline = cpwl.PiecewiseLinearProfile(torch.tensor(cfg.knots), torch.tensor(cfg.vals))
        return (spline.eval_relu_form, lambda M, sgn: cpwl.spline_map_sign(M, spline, sgn))
    raise ValueError(f"unknown profile {cfg.profile!r}")


def run_cpwl_convergence(cfg: CPWLOperatorConfig) -> dict[str, dict[int, torch.Tensor]]:
    """Runs every degree in cfg.degrees on one matrix; returns
    {"error": {D: e_0..e_kmax}}, the relative Frobenius error of the
    decomposition-free CPWL operator (cfg.profile's sign form, at k
    Newton-Schulz iterations shared by every internal sign call) against the
    exact operator read off a signed SVD. Usage:
    run_cpwl_convergence(CPWLOperatorConfig(profile="clip", alpha=-0.5, beta=0.5)).
    """
    M, _ = orbit_tools.make_instance(cfg.m, cfg.n, cfg.rank, cfg.smin, cfg.seed, device=cfg.device)
    scalar_fn, sign_form = resolve_profile(cfg)
    Y_exact = metrics.op_svd(M, scalar_fn)
    Md = M.to(cfg.dtype)
    tol = orbit_tools.default_tol(cfg.dtype) if cfg.tol is None else cfg.tol

    error = {}
    for D in cfg.degrees:
        e = torch.empty(cfg.k_max + 1, dtype=torch.float64)
        for k in range(cfg.k_max + 1):
            sgn = sign_map.make_sgn_ns(D, k, tol=tol)
            Y = sign_form(Md, sgn).to(torch.float64)
            e[k] = metrics.relative_frobenius_error(Y, Y_exact)
        error[D] = e
    return {"error": error}


def run_cpwl_spectrum(cfg: CPWLOperatorConfig) -> dict[str, object]:
    """Runs cfg.D_show on one matrix at every k in cfg.k_show; returns
    {"sigma": singular values of M, "target": the exact scalar profile at
    sigma, "coords": {k: spectral coordinates of the operator's output at k
    Newton-Schulz iterations}}. Usage:
    run_cpwl_spectrum(CPWLOperatorConfig(profile="clip", D_show=3, k_show=[1, 2, 4, 8])).
    """
    M, _ = orbit_tools.make_instance(cfg.m, cfg.n, cfg.rank, cfg.smin, cfg.seed, device=cfg.device)
    scalar_fn, sign_form = resolve_profile(cfg)
    U, sigma, V = metrics.reference_svd(M)
    target = scalar_fn(sigma)
    Md = M.to(cfg.dtype)
    tol = orbit_tools.default_tol(cfg.dtype) if cfg.tol is None else cfg.tol

    coords = {}
    for k in cfg.k_show:
        sgn = sign_map.make_sgn_ns(cfg.D_show, k, tol=tol)
        Y = sign_form(Md, sgn).to(torch.float64)
        coords[k] = metrics.spectral_coordinates(Y, U, V)
    return {"sigma": sigma, "target": target, "coords": coords}
