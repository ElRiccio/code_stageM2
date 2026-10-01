"""Polynomial comparison experiment: run the Muon quintic, the Björck quintic and
the max-derivative quintic on one matrix, and record the error against the exact
sign and the singular values after every step. It also measures a CPWL
operator built on a fixed number of steps of one of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ns_core import metrics, ns_iteration, orbit_tools, profiles, sign_map
from experiments import cpwl_operator


@dataclass
class PolynomialComparisonConfig:
    """
    m, n: matrix shape
    rank: see orbit_tools.resolve_rank (None = full)
    cond: largest over smallest nonzero singular value
    sigma_max: largest singular value (the iteration runs on M / sigma_max)
    polynomials: odd polynomial coefficients by name (None = named_polynomials())
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
    polynomials: dict[str, torch.Tensor] | None = None
    k_max: int = 15
    eps: float | None = None
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    seed: int = 0


def named_polynomials() -> dict[str, torch.Tensor]:
    """
    Returns: coefficients (ascending, for x, x^3, x^5) of "Muon", "Björck", "Max derivative"
    """
    return {
        "Muon": torch.tensor([3.4445, -4.7750, 2.0315], dtype=torch.float64),
        "Björck": ns_iteration.bpoly_coeffs(2),
        "Max derivative": profiles.quintic_coeffs(
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor(profiles.QUINTIC_BMAX, dtype=torch.float64),
        ),
    }


def run_polynomial_comparison(cfg: PolynomialComparisonConfig) -> dict[str, object]:
    """
    cfg: settings
    Returns: {"error": {name: spectral-norm error at steps 0..k_max}, "sigma": singular
    values of M / sigma_max, "target": exact sign at sigma, "coords": {name: {k: spectrum after k steps, k = 0..k_max}}}
    Note: the error is held once it reaches eps, so a polynomial that does not converge
    to the sign never freezes
    """
    polynomials = named_polynomials() if cfg.polynomials is None else cfg.polynomials
    M, N = orbit_tools.make_instance(
        cfg.m, cfg.n, cfg.rank, cfg.cond, cfg.seed, sigma_max=cfg.sigma_max, normalize=True, device=cfg.device
    )
    U, sigma, V = metrics.reference_svd(M)
    target = (sigma > metrics.numerical_rank_tol(M, sigma)).to(sigma.dtype)

    error, coords = {}, {}
    for name, coeffs in polynomials.items():
        orbit, error[name] = orbit_tools.run_orbit(
            M, N, None, cfg.k_max, cfg.dtype, cfg.eps, coeffs=coeffs
        )
        coords[name] = {k: metrics.spectral_coordinates(orbit[k], U, V) for k in range(cfg.k_max + 1)}
    return {"error": error, "sigma": sigma, "target": target, "coords": coords}


def run_quintic_cpwl(
    cfg: cpwl_operator.CPWLOperatorConfig, coeffs: torch.Tensor | None = None, n_iters: int = 5
) -> dict[str, float]:
    """
    cfg: CPWL settings (profile, parameters, matrix, dtype, device, seed)
    coeffs: odd polynomial for the sign map (None = the max-derivative quintic)
    n_iters: steps of every internal sign call
    Returns: {"error": relative Frobenius error of the operator against the exact one}
    """
    if coeffs is None:
        coeffs = named_polynomials()["Max derivative"]
    M, _ = orbit_tools.make_instance(
        cfg.m, cfg.n, cfg.rank, cfg.cond, cfg.seed, sigma_max=cfg.sigma_max, device=cfg.device
    )
    scalar_fn, sign_form = cpwl_operator.resolve_profile(cfg)
    Y_exact = cpwl_operator.spectral_reference(M, scalar_fn)[4]
    sgn = sign_map.make_sgn_ns(None, n_iters, coeffs=coeffs)
    Y = sign_form(M.to(cfg.dtype), sgn).to(torch.float64)
    return {"error": float(metrics.relative_frobenius_error(Y, Y_exact))}


def run_cpwl_comparison(
    cfg: cpwl_operator.CPWLOperatorConfig,
    polynomials: dict[str, torch.Tensor] | None = None,
    k_max: int = 15,
) -> dict[str, object]:
    """
    cfg: CPWL settings (profile, parameters, matrix, dtype, device, seed, eps)
    polynomials: odd polynomial coefficients by name (None = named_polynomials())
    k_max: steps
    Returns: {"error": {name: relative Frobenius error at k = 0..k_max}, "sigma": singular
    values of M, "target": exact profile at sigma, "coords": {name: {k: output spectrum}}}
    Note: every internal sign call uses the same k steps; once the error reaches cfg.eps the
    remaining steps are skipped and the error and spectrum are held
    """
    polynomials = named_polynomials() if polynomials is None else polynomials
    M, _ = orbit_tools.make_instance(
        cfg.m, cfg.n, cfg.rank, cfg.cond, cfg.seed, sigma_max=cfg.sigma_max, device=cfg.device
    )
    scalar_fn, sign_form = cpwl_operator.resolve_profile(cfg)
    U, sigma, V, target, Y_exact = cpwl_operator.spectral_reference(M, scalar_fn)
    Md = M.to(cfg.dtype)
    eps = orbit_tools.default_eps(cfg.dtype) if cfg.eps is None else cfg.eps

    error, coords = {}, {}
    for name, coeffs in polynomials.items():
        e = torch.empty(k_max + 1, dtype=torch.float64)
        coords[name] = {}
        for k in range(k_max + 1):
            Y = sign_form(Md, sign_map.make_sgn_ns(None, k, coeffs=coeffs)).to(torch.float64)
            e[k] = metrics.relative_frobenius_error(Y, Y_exact)
            coords[name][k] = metrics.spectral_coordinates(Y, U, V)
            if e[k] <= eps:
                e[k + 1 :] = e[k]
                for j in range(k + 1, k_max + 1):
                    coords[name][j] = coords[name][k]
                break
        error[name] = e
    return {"error": error, "sigma": sigma, "target": target, "coords": coords}
