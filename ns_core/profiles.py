"""Admissible-profile constants: the quantities that describe how p_D
behaves near the fixed point x = 1, and the basin-of-attraction / iteration
count results built from the truncated series B_D = sum_{j<=D} c_j t^j, the
degree-D truncation of (1 - t)^{-1/2}.

These are the quantities Chapter 4's degree-D trade-off (convergence order
vs. basin of attraction vs. arithmetic cost) is stated in terms of, and
what a sweep over D validates numerically.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ns_core import ns_iteration

_CONST_CACHE: dict[tuple[int, int], tuple[float, float, float]] = {}


def admissible_constants(D: int, n_grid: int) -> tuple[float, float, float]:
    """(mu_D, rho_D, lambda_D):
      mu_D      sup_{x in [0,1]} |psi_D(x)|, the deflated-residual bound;
      rho_D     mu_D^(-1/D), the radius of the ball around 1 that p_D maps
                into itself with contraction;
      lambda_D  T_D(1 - rho_D), the one-step growth rate used by the
                burn-in bound.

    Estimated on a grid of `n_grid` points on [0, 1]; cached per (D, n_grid).
    """
    if D < 1:
        raise ValueError("the constants are defined for D >= 1")
    key = (D, n_grid)
    if key in _CONST_CACHE:
        return _CONST_CACHE[key]

    x = torch.linspace(0.0, 1.0, n_grid, dtype=torch.float64)
    psi = ns_iteration.deflation_coeffs(D, dtype=torch.float64)
    vals = torch.zeros_like(x)
    for c in reversed(psi):
        vals = vals * x + c
    mu = float(torch.max(torch.abs(vals)))
    rho = mu ** (-1.0 / D)
    lam = float(ns_iteration.t_poly_eval(torch.tensor(1.0 - rho, dtype=torch.float64), D))

    _CONST_CACHE[key] = (mu, rho, lam)
    return mu, rho, lam


def burnin_bound(u0: float, D: int, n_grid: int) -> int:
    """ceil( log((1 - rho_D) / u0) / log(lambda_D) ): the number of steps the
    theory guarantees suffice to bring u0 into the contraction ball of
    radius rho_D. Returns 0 if u0 is already inside the ball.
    """
    _, rho, lam = admissible_constants(D, n_grid)
    if u0 >= 1.0 - rho:
        return 0
    return int(math.ceil(math.log((1.0 - rho) / u0) / math.log(lam)))


def truncated_series_coeffs(D: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Ascending coefficients c_0, ..., c_D of B_D, the degree-D truncation
    of the binomial series (1 - t)^{-1/2}."""
    if D < 0:
        raise ValueError("D must be nonnegative")
    coeffs = [ns_iteration.central_binomial(j) / 4.0**j for j in range(D + 1)]
    return torch.tensor(coeffs, dtype=dtype)


def truncated_series_eval(t: torch.Tensor, D: int) -> torch.Tensor:
    """B_D(t) = sum_{j <= D} c_j t^j."""
    coeffs = truncated_series_coeffs(D, dtype=t.dtype).to(device=t.device)
    out = torch.zeros_like(t)
    for c in reversed(coeffs):
        out = out * t + c
    return out


def basin_radius(D: int) -> tuple[float, float]:
    """(R_D, t_D): the basin-of-attraction radius and the largest negative
    real root t_D of Xi_D, where Xi_D = B_D for D odd and Xi_D = B_D with
    its constant term dropped for D even.

    The root itself is found with numpy.roots (via the companion matrix):
    finding roots of a fixed, low-degree polynomial is a one-off dense
    eigenvalue problem with no matrix structure to exploit, so this stays
    on NumPy rather than going through torch.linalg for it.
    """
    if D < 1:
        raise ValueError("the radius is defined for D >= 1")
    c = truncated_series_coeffs(D, dtype=torch.float64).tolist()
    xi = c if D % 2 == 1 else c[1:]  # ascending coefficients of Xi_D
    roots = np.roots(xi[::-1])  # numpy.roots wants them descending
    real = roots[np.abs(roots.imag) < 1e-10].real
    neg = real[real < 0.0]
    if neg.size == 0:
        raise RuntimeError(f"no negative root found for D={D}")
    t_D = float(np.max(neg))  # largest negative real root
    return math.sqrt(1.0 - t_D), t_D


def iteration_count_bound(D: int, theta: float, eps: float) -> int:
    """K_D(theta, eps): the number of NS steps the basin-of-attraction bound
    guarantees suffice to reach relative error eps, starting from relative
    distance theta to the target, at degree D. Requires theta, eps in (0, 1).
    """
    theta, eps = float(theta), float(eps)
    if not 0.0 < theta < 1.0 or not 0.0 < eps < 1.0:
        raise ValueError("require theta, eps in (0, 1)")
    ratio = math.log(1.0 / eps) / math.log(1.0 / theta)
    if ratio <= 1.0:
        return 0
    return int(math.ceil(math.log(ratio) / math.log(D + 1)))
