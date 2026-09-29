"""The Newton-Schulz iteration: the odd polynomial p_D, and its repeated
application to numbers (scalar side) and to matrices (matrix side).

Orbit functions take either a degree D or a coefficient tensor `coeffs` for any
odd polynomial, e.g. the quintics from `profiles.quintic_coeffs`.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

_COEFF_CACHE: dict[tuple[int, torch.dtype, torch.device], torch.Tensor] = {}


@lru_cache(maxsize=None)
def _t_d_coeffs(D: int) -> tuple[float, ...]:
    """
    D: degree
    Returns: series coefficients as Python floats
    """
    return tuple(central_binomial(j) / 4.0**j for j in range(D + 1))


# ----------------------------------------------------------------------------
# Coefficients of p_D
# ----------------------------------------------------------------------------


def central_binomial(j: int) -> int:
    """
    j: index
    Returns: central binomial coefficient
    """
    return math.comb(2 * j, j)


def bpoly_coeffs(
    D: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    D: degree
    dtype, device: output type and place
    Returns: coefficients of p_D
    Note: computed in float64, cached
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
    """
    D: degree
    Returns: error constant of the (D+1)-order convergence
    """
    if D < 0:
        raise ValueError("D must be nonnegative")
    return central_binomial(D + 1) / 2.0 ** (D + 1)


# ----------------------------------------------------------------------------
# Scalar reading, on (-1, 1)
# ----------------------------------------------------------------------------


def t_poly_eval(x: torch.Tensor, D: int) -> torch.Tensor:
    """
    x: points in [-1, 1]
    D: degree
    Returns: p_D(x) / x, same shape as x
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
    """
    x: points
    D: degree
    Returns: p_D(x)
    """
    return x * t_poly_eval(x, D)


def odd_poly_eval(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    x: points
    coeffs: odd polynomial coefficients
    Returns: polynomial value, same shape as x
    """
    coeffs = coeffs.to(device=x.device, dtype=x.dtype)
    x2 = x * x
    out = torch.zeros_like(x)
    for a in reversed(coeffs):
        out = out * x2 + a
    return x * out


def scalar_orbit(
    u0: torch.Tensor,
    D: int | None,
    n_iters: int,
    *,
    coeffs: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    u0: starting points
    D: degree (None if coeffs given)
    n_iters: steps
    coeffs: custom odd polynomial
    Returns: all iterates, shape (n_iters + 1, *u0.shape)
    """
    if n_iters < 0:
        raise ValueError("n_iters must be nonnegative")
    out = torch.empty((n_iters + 1, *u0.shape), dtype=u0.dtype, device=u0.device)
    out[0] = u0
    u = u0
    for k in range(n_iters):
        u = bpoly_eval(u, D) if coeffs is None else odd_poly_eval(u, coeffs)
        out[k + 1] = u
    return out


def log10_error_orbit(
    u0: torch.Tensor | float,
    D: int | None,
    n_iters: int,
    *,
    coeffs: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    u0: starting points in (0, 1)
    D: degree (None if coeffs given)
    n_iters: steps
    coeffs: custom odd polynomial
    Returns: log10 of the distance to 1 along the orbit
    Note: floored at machine epsilon; a float u0 gives a float64 CPU result
    """
    u0 = torch.as_tensor(u0, dtype=torch.float64) if not torch.is_tensor(u0) else u0
    if not bool(((u0 > 0.0) & (u0 < 1.0)).all()):
        raise ValueError("u0 must lie in the open interval (0, 1)")
    u = scalar_orbit(u0, D, n_iters, coeffs=coeffs)
    floor = torch.finfo(u.dtype).eps
    return torch.log10(torch.clamp((1.0 - u).abs(), min=floor))


# ----------------------------------------------------------------------------
# Matrix reading
# ----------------------------------------------------------------------------


def ns_step_matrix(X: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    X: current iterate
    coeffs: polynomial coefficients
    Returns: next iterate
    """
    G = X @ X.mH
    out = coeffs[0] * X
    Pk = X
    for j in range(1, coeffs.numel()):
        Pk = G @ Pk  # (X X^T)^j X, one product per degree
        out = out + coeffs[j] * Pk
    return out


def gram_matrix(X: torch.Tensor) -> torch.Tensor:
    """
    X: input matrix
    Returns: Gram matrix on the smaller side
    """
    return X.mH @ X if X.shape[-2] >= X.shape[-1] else X @ X.mH


def ns_step_gram(
    X: torch.Tensor, coeffs: torch.Tensor, gram: torch.Tensor | None = None
) -> torch.Tensor:
    """
    X: current iterate
    coeffs: polynomial coefficients
    gram: precomputed gram_matrix(X), skips that product
    Returns: next iterate, same as ns_step_matrix
    Note: cheaper than ns_step_matrix; gram is not modified
    """
    D = coeffs.numel() - 1
    if D == 0:
        return coeffs[0] * X
    tall = X.shape[-2] >= X.shape[-1]
    G = gram_matrix(X) if gram is None else gram
    Q = coeffs[D] * G
    Q.diagonal(dim1=-2, dim2=-1).add_(coeffs[D - 1])  # Q = a[D] G + a[D-1] I
    for j in range(D - 2, -1, -1):
        Q = Q @ G
        Q.diagonal(dim1=-2, dim2=-1).add_(coeffs[j])  # Q = Q G + a[j] I
    return X @ Q if tall else Q @ X


def ns_orbit_matrix(
    M: torch.Tensor,
    D: int | None,
    n_iters: int,
    *,
    coeffs: torch.Tensor | None = None,
    scale: torch.Tensor | float | None = None,
    tol: float | None = None,
) -> list[torch.Tensor]:
    """
    M: input matrix
    D: degree (None if coeffs given)
    n_iters: steps
    coeffs: custom odd polynomial
    scale: divisor for M (default: spectral norm)
    tol: early stop when a step changes less than this
    Returns: list of n_iters + 1 iterates
    Note: after an early stop the last iterate is repeated
    """
    if n_iters < 0:
        raise ValueError("n_iters must be nonnegative")
    if coeffs is None:
        coeffs = bpoly_coeffs(D, dtype=M.dtype, device=M.device)
    else:
        coeffs = coeffs.to(device=M.device, dtype=M.dtype)
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
