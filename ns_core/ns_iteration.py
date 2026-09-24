"""The Bjorck-Bowie polynomial family p_D(x) = x B_D(1 - x^2) and the
Newton-Schulz recursion it drives, on the scalar line and on matrices.

Scalar side: coefficients of p_D, its evaluation in the residual variable
t = 1 - x^2, orbits u_{k+1} = p_D(u_k) and the asymptotic error constant.
Matrix side: one step of the odd matrix polynomial and the orbit
X_{k+1} = Phi(X_k) started from a rescaled M.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

_COEFF_CACHE: dict[tuple[int, torch.dtype, torch.device], torch.Tensor] = {}


@lru_cache(maxsize=None)
def _t_d_coeffs(D: int) -> tuple[float, ...]:
    """Python-float coefficients c_j = C(2j, j) / 4^j, j = 0..D."""
    return tuple(central_binomial(j) / 4.0**j for j in range(D + 1))


# ----------------------------------------------------------------------------
# Coefficients of p_D
# ----------------------------------------------------------------------------


def central_binomial(j: int) -> int:
    """Central binomial coefficient C(2j, j).

    Usage: central_binomial(3)  # 20
    """
    return math.comb(2 * j, j)


def bpoly_coeffs(
    D: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Coefficients a[0..D] of p_D(x) = sum_j a[j] x^(2j+1), cached per
    (D, dtype, device). The expansion is carried out in double precision and
    the result is returned in `dtype` on `device`.

    Usage: a = bpoly_coeffs(2, dtype=M.dtype, device=M.device)
    """
    if D < 0:
        raise ValueError("D must be nonnegative")
    key = (D, dtype, torch.device(device))
    if key in _COEFF_CACHE:
        return _COEFF_CACHE[key]
    a = [0.0] * (D + 1)
    for j in range(D + 1):
        cj = central_binomial(j) / 4.0**j
        for i in range(j + 1):  # binomial expansion of (1 - x^2)^j
            a[i] += cj * math.comb(j, i) * (-1) ** i
    coeffs = torch.tensor(a, dtype=dtype, device=device)
    _COEFF_CACHE[key] = coeffs
    return coeffs


def asymptotic_error_constant(D: int) -> float:
    """kappa_D = C(2D+2, D+1) / 2^(D+1): the constant in the order-(D+1)
    convergence u_k -> 1 of p_D.

    Usage: asymptotic_error_constant(2)  # 2.5
    """
    if D < 0:
        raise ValueError("D must be nonnegative")
    return central_binomial(D + 1) / 2.0 ** (D + 1)


# ----------------------------------------------------------------------------
# Scalar reading, on (-1, 1)
# ----------------------------------------------------------------------------


def t_poly_eval(x: torch.Tensor, D: int) -> torch.Tensor:
    """B_D(1 - x^2) = sum_{j<=D} c_j (1 - x^2)^j, summed in t = 1 - x^2 where
    every term is nonnegative on [-1, 1]. Shape, dtype and device follow `x`.

    Usage: t_poly_eval(torch.linspace(0, 1, 5), D=2)
    """
    if D < 0:
        raise ValueError("D must be nonnegative")
    coeffs = torch.tensor(_t_d_coeffs(D), dtype=x.dtype, device=x.device)
    t = 1.0 - x * x
    out = torch.zeros_like(x)
    tp = torch.ones_like(x)
    for c in coeffs:
        out = out + c * tp
        tp = tp * t
    return out


def bpoly_eval(x: torch.Tensor, D: int) -> torch.Tensor:
    """p_D(x) = x B_D(1 - x^2), evaluated through the residual variable.

    Usage: bpoly_eval(torch.tensor([0.3, 0.9]), D=2)
    """
    return x * t_poly_eval(x, D)


def scalar_orbit(u0: torch.Tensor, D: int, n_iters: int) -> torch.Tensor:
    """Orbit u_0, ..., u_K of u_{k+1} = p_D(u_k), as a tensor of shape
    (n_iters + 1, *u0.shape).

    Usage: scalar_orbit(torch.tensor([0.1, 0.5]), D=2, n_iters=8)
    """
    if n_iters < 0:
        raise ValueError("n_iters must be nonnegative")
    out = torch.empty((n_iters + 1, *u0.shape), dtype=u0.dtype, device=u0.device)
    out[0] = u0
    u = u0
    for k in range(n_iters):
        u = bpoly_eval(u, D)
        out[k + 1] = u
    return out


# ----------------------------------------------------------------------------
# Matrix reading
# ----------------------------------------------------------------------------


def ns_step_matrix(X: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """One step of the odd matrix polynomial Phi(X) = sum_j a[j] (X X^T)^j X,
    with one matrix product per degree. `coeffs` holds a[0..D] on X's device.

    Usage: ns_step_matrix(X, bpoly_coeffs(2, dtype=X.dtype, device=X.device))
    """
    G = X @ X.mH
    out = coeffs[0] * X
    Pk = X
    for j in range(1, coeffs.numel()):
        Pk = G @ Pk  # (X X^T)^j X, one product per degree
        out = out + coeffs[j] * Pk
    return out


def ns_orbit_matrix(
    M: torch.Tensor,
    D: int,
    n_iters: int,
    *,
    scale: torch.Tensor | float | None = None,
    tol: float | None = None,
) -> list[torch.Tensor]:
    """Iterates X_0, ..., X_K of X_{k+1} = Phi(X_k) with X_0 = M / scale,
    where `scale` defaults to the spectral norm of M. The whole orbit shares
    one scale, so every iterate is compared with the same msgn(M) target.

    With `tol`, iteration stops once ||X_{k+1} - X_k||_F < tol and the last
    iterate is repeated, so the list always has n_iters + 1 entries.

    Usage: orbit = ns_orbit_matrix(M, D=2, n_iters=10)
    """
    if n_iters < 0:
        raise ValueError("n_iters must be nonnegative")
    coeffs = bpoly_coeffs(D, dtype=M.dtype, device=M.device)
    if scale is None:
        scale = torch.linalg.matrix_norm(M, ord=2)
    scale = torch.as_tensor(scale, dtype=M.dtype, device=M.device)
    if scale <= 0:
        return [torch.zeros_like(M) for _ in range(n_iters + 1)]

    X = M / scale
    out = [X]
    stopped = False
    for _ in range(n_iters):
        if not stopped:
            X_prev = X
            X = ns_step_matrix(X, coeffs)
            if tol is not None and torch.linalg.norm(X - X_prev) < tol:
                stopped = True
        out.append(X)
    return out
