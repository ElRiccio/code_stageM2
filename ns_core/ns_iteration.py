"""The degree-(2D+1) odd polynomial family p_D and the Newton-Schulz-style
recursion it drives, read both as a scalar map on (-1, 1) and as an odd
matrix polynomial acting on the singular values of a rescaled matrix.

Coefficients are cached per degree D, since a handful of degrees are reused
across many experiments (degree comparisons, repeated random trials) and
recomputing the binomial expansion each call would be wasted work.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

_COEFF_CACHE: dict[tuple[int, torch.dtype], torch.Tensor] = {}
_DEFLATION_CACHE: dict[tuple[int, torch.dtype], torch.Tensor] = {}


@lru_cache(maxsize=None)
def _t_d_coeffs(D: int) -> tuple[float, ...]:
    """Python-float coefficients C(2j,j) / 4^j, j = 0..D, of T_D. Cached
    since bpoly_eval and log10_error_orbit both build on them."""
    return tuple(central_binomial(j) / 4.0**j for j in range(D + 1))


def _p_d_scalar(x: float, D: int) -> float:
    """Plain-float evaluation of p_D(x), for the scalar error-orbit tracker
    where building a 0-d tensor per step would just add overhead."""
    t = 1.0 - x * x
    out = 0.0
    tp = 1.0
    for c in _t_d_coeffs(D):
        out += c * tp
        tp *= t
    return x * out


def _polyval_ascending(x: float, coeffs: list[float]) -> float:
    """Horner evaluation of sum_i coeffs[i] x^i, coeffs ascending."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * x + c
    return result


def _divide_by_x_minus_1(coeffs: list[float]) -> tuple[list[float], float]:
    """Synthetic division of a polynomial (ascending coefficients) by
    (x - 1). Returns (quotient, remainder); remainder should be ~0 whenever
    x = 1 is genuinely a root, which deflation_coeffs relies on."""
    descending = list(reversed(coeffs))
    quotient_and_remainder = [descending[0]]
    for c in descending[1:]:
        quotient_and_remainder.append(c + 1.0 * quotient_and_remainder[-1])
    remainder = quotient_and_remainder[-1]
    quotient = list(reversed(quotient_and_remainder[:-1]))
    return quotient, remainder


# ----------------------------------------------------------------------------
# Coefficients of p_D
# ----------------------------------------------------------------------------


def central_binomial(j: int) -> int:
    """Central binomial coefficient C(2j, j)."""
    return math.comb(2 * j, j)


def bpoly_coeffs(D: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Coefficients a[0..D] of p_D(x) = sum_j a[j] x^(2j+1).

    Computed in float64 regardless of the dtype the iteration itself will
    run in: these coefficients are reused at every step, so their own
    rounding error should not be the bottleneck. Cached per degree.
    """
    if D < 0:
        raise ValueError("D must be nonnegative")
    key = (D, dtype)
    if key in _COEFF_CACHE:
        return _COEFF_CACHE[key]
    a = [0.0] * (D + 1)
    for j in range(D + 1):
        cj = central_binomial(j) / 4.0**j
        for i in range(j + 1):  # binomial expansion of (1 - x^2)^j
            a[i] += cj * math.comb(j, i) * (-1) ** i
    coeffs = torch.tensor(a, dtype=dtype)
    _COEFF_CACHE[key] = coeffs
    return coeffs


def deflation_coeffs(D: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Coefficients of psi_D, where p_D(x) - 1 = (x - 1)^(D+1) psi_D(x).

    Used to evaluate the error orbit stably once |1 - u_k| underflows to
    where p_D(u_k) - 1 would cancel directly.
    """
    if D < 0:
        raise ValueError("D must be nonnegative")
    key = (D, dtype)
    if key in _DEFLATION_CACHE:
        return _DEFLATION_CACHE[key]
    a = bpoly_coeffs(D, dtype=torch.float64).tolist()
    full = [0.0] * (2 * D + 2)  # ascending monomial coeffs of p_D, even entries zero
    for j, val in enumerate(a):
        full[2 * j + 1] = val
    full[0] -= 1.0  # now p_D(x) - 1
    q = full
    for _ in range(D + 1):  # (x - 1) is a root of multiplicity D + 1
        q, _ = _divide_by_x_minus_1(q)
    coeffs = torch.tensor(q, dtype=dtype)
    _DEFLATION_CACHE[key] = coeffs
    return coeffs


def asymptotic_error_constant(D: int) -> float:
    """|psi_D(1)| = C(2D+2, D+1) / 2^(D+1): the asymptotic error constant of
    the order-(D+1) convergence rate."""
    if D < 0:
        raise ValueError("D must be nonnegative")
    return central_binomial(D + 1) / 2.0 ** (D + 1)


# ----------------------------------------------------------------------------
# Scalar reading, on (-1, 1)
# ----------------------------------------------------------------------------


def t_poly_eval(x: torch.Tensor, D: int) -> torch.Tensor:
    """T_D(x) = sum_j C(2j,j) (1 - x^2)^j / 4^j.

    Every term is nonnegative on [-1, 1] (summed in t = 1 - x^2), so nothing
    cancels near x = 1; p_D(x) = x T_D(x) is built on this for exactly that
    reason (see bpoly_eval), and T_D itself is the quantity the burn-in
    growth rate lambda_D is evaluated from (ns_core.profiles).
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
    """p_D(x) = x T_D(x).

    Preferred over direct Horner-in-x evaluation of the monomial
    coefficients: T_D sums in t = 1 - x^2, where every term is nonnegative
    on [-1, 1], so nothing cancels near x = 1 — exactly the regime the NS
    iteration spends most of its steps in.
    """
    return x * t_poly_eval(x, D)


def scalar_orbit(u0: torch.Tensor, D: int, n_iters: int) -> torch.Tensor:
    """Orbit u_0, ..., u_K of u_{k+1} = p_D(u_k).

    Returns a tensor of shape (n_iters + 1, *u0.shape).
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


def log10_error_orbit(u0: float, D: int, n_iters: int, switch: float = 1e-8) -> torch.Tensor:
    """log10 |1 - u_k| for u0 in (0, 1), switching to the deflated form
    (D+1) log10|1 - u_k| + log10|psi_D(u_k)| once |1 - u_k| <= `switch`, to
    stay accurate past the point where direct subtraction would cancel.

    This is the quantity a numerical check of the theoretical order-(D+1)
    convergence rate is built on: the slope of this curve, or of the
    corresponding log-log error-ratio plot, should approach D+1.
    """
    u0 = float(u0)
    if not 0.0 < u0 < 1.0:
        raise ValueError("u0 must lie in the open interval (0, 1)")
    if n_iters < 0:
        raise ValueError("n_iters must be nonnegative")

    psi = deflation_coeffs(D, dtype=torch.float64).tolist()
    log_psi_1 = math.log10(asymptotic_error_constant(D))

    L = [0.0] * (n_iters + 1)
    u = u0
    e = 1.0 - u
    L[0] = math.log10(e)
    for k in range(n_iters):
        if e > switch:
            u_next = min(max(_p_d_scalar(u, D), 0.0), 1.0)
            e_next = 1.0 - u_next
            if e_next > switch:
                L[k + 1] = math.log10(e_next)
            else:
                val = abs(_polyval_ascending(u, psi))
                L[k + 1] = (D + 1) * L[k] + math.log10(val)
            u, e = u_next, e_next
        else:
            L[k + 1] = (D + 1) * L[k] + log_psi_1
    return torch.tensor(L, dtype=torch.float64)


# ----------------------------------------------------------------------------
# Matrix reading
# ----------------------------------------------------------------------------


def ns_step_matrix(X: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """One step of the odd matrix polynomial Phi(X) = sum_j a[j] (X X^T)^j X.

    Built with one matrix product per degree (Horner in the matrix G = X X^T)
    rather than forming (X X^T)^j explicitly at each j.
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
    """Iterates X_0, ..., X_K of the matrix recursion, started at M rescaled
    to unit spectral norm (or to `scale` if given explicitly).

    The whole orbit shares one pre-scaling, since Sgn(M / beta) = Sgn(M) for
    any beta > 0: comparing every iterate against a single target is valid
    exactly because of that invariance, not by coincidence.

    If `tol` is given, iterating stops early once the Frobenius step size
    ||X_{k+1} - X_k||_F drops below it (past that point, the null-space
    directions of a rank-deficient M sit at an unstable fixed point of Phi,
    so further steps mostly feed floating-point noise back into itself).
    The returned list still has n_iters + 1 entries: once stopped, the last
    iterate is repeated so the result stays indexable exactly like the
    tol=None case.
    """
    if n_iters < 0:
        raise ValueError("n_iters must be nonnegative")
    coeffs = bpoly_coeffs(D, dtype=M.dtype).to(device=M.device)
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
