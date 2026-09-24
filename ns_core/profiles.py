"""Profiles built from the series B_infinity(t) = (1 - t)^(-1/2) in the
residual variable t = 1 - x^2.

Truncated family: the coefficients c_j of B_D, its evaluation, the basin
radius R_D = sqrt(1 - t_D) and the iteration count K_D.
Admissible quintics p(x) = x (1 + r1 t + r2 t^2): the x-polynomial
coefficients, the constant bmax, the slopes at 0 and 1, the extremal slope,
and the order of convergence with its constant. Coefficient tensors have the
layout of `ns_iteration.bpoly_coeffs`, so `ns_iteration.ns_step_matrix` runs
them directly.
"""

from __future__ import annotations

import math

import torch

from ns_core import ns_iteration

# bmax = (51 sqrt(17) - 107) / 64, the largest r2 of an admissible quintic
QUINTIC_BMAX: float = (51.0 * math.sqrt(17.0) - 107.0) / 64.0
# max p'(0) = 1 + bmax = (51 sqrt(17) - 43) / 64, attained at (r1, r2) = (0, bmax)
QUINTIC_MAX_SLOPE: float = 1.0 + QUINTIC_BMAX


# ----------------------------------------------------------------------------
# Truncated series B_D
# ----------------------------------------------------------------------------


def truncated_series_coeffs(
    D: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Ascending coefficients c_0, ..., c_D of B_D, the degree-D truncation
    of (1 - t)^(-1/2).

    Usage: c = truncated_series_coeffs(3, dtype=t.dtype, device=t.device)
    """
    if D < 0:
        raise ValueError("D must be nonnegative")
    coeffs = [ns_iteration.central_binomial(j) / 4.0**j for j in range(D + 1)]
    return torch.tensor(coeffs, dtype=dtype, device=device)


def truncated_series_eval(t: torch.Tensor, D: int) -> torch.Tensor:
    """B_D(t) = sum_{j<=D} c_j t^j by Horner's rule; shape, dtype and device
    follow `t`.

    Usage: truncated_series_eval(torch.linspace(-1, 1, 5), D=2)
    """
    coeffs = truncated_series_coeffs(D, dtype=t.dtype, device=t.device)
    out = torch.zeros_like(t)
    for c in reversed(coeffs):
        out = out * t + c
    return out


def basin_radius(D: int) -> tuple[float, float]:
    """(R_D, t_D): the basin radius R_D = sqrt(1 - t_D), where t_D is the
    largest negative real root of Xi_D. Xi_D is B_D for odd D and B_D without
    its constant term for even D. The roots are the eigenvalues of the
    companion matrix of Xi_D.

    Usage: R, t = basin_radius(3)  # R ~ 1.5893
    """
    if D < 1:
        raise ValueError("the radius is defined for D >= 1")
    c = truncated_series_coeffs(D, dtype=torch.float64)
    xi = c if D % 2 == 1 else c[1:]  # ascending coefficients of Xi_D
    n = xi.numel() - 1
    companion = torch.zeros((n, n), dtype=torch.float64)
    companion[1:, :-1] = torch.eye(n - 1, dtype=torch.float64)
    companion[:, -1] = -xi[:-1] / xi[-1]
    roots = torch.linalg.eigvals(companion)
    real = roots.real[roots.imag.abs() < 1e-10]
    neg = real[real < 0.0]
    if neg.numel() == 0:
        raise RuntimeError(f"no negative root found for D={D}")
    t_D = float(neg.max())
    return math.sqrt(1.0 - t_D), t_D


def iteration_count_bound(D: int, theta: float, eps: float) -> int:
    """K_D: the number of steps of p_D after which |u_k - 1| <= eps, for a
    start with residual theta = 1 - u_0^2. Both theta and eps lie in (0, 1).

    Usage: iteration_count_bound(2, theta=1 - 0.01**2, eps=1e-6)
    """
    theta, eps = float(theta), float(eps)
    if not 0.0 < theta < 1.0 or not 0.0 < eps < 1.0:
        raise ValueError("require theta, eps in (0, 1)")
    ratio = math.log(1.0 / eps) / math.log(1.0 / theta)
    if ratio <= 1.0:
        return 0
    return int(math.ceil(math.log(ratio) / math.log(D + 1)))


# ----------------------------------------------------------------------------
# Admissible quintics p(x) = x (1 + r1 t + r2 t^2), t = 1 - x^2
# ----------------------------------------------------------------------------


def quintic_coeffs(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """Coefficients a[0..2] of p(x) = a0 x + a1 x^3 + a2 x^5 for the residual
    coefficients (r1, r2): a = (1 + r1 + r2, -(r1 + 2 r2), r2). Inputs
    broadcast; the result has shape (..., 3) on the inputs' device and dtype.

    Usage: quintic_coeffs(torch.tensor(0.5), torch.tensor(0.375))  # p_2
    """
    r1, r2 = torch.broadcast_tensors(r1, r2)
    return torch.stack([1.0 + r1 + r2, -(r1 + 2.0 * r2), r2], dim=-1)


def quintic_slope_origin(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """p'(0) = 1 + r1 + r2, the factor applied to the smallest singular
    values in the first step.

    Usage: quintic_slope_origin(torch.tensor(0.0), torch.tensor(QUINTIC_BMAX))
    """
    return 1.0 + r1 + r2


def quintic_slope_one(r1: torch.Tensor) -> torch.Tensor:
    """p'(1) = 1 - 2 r1.

    Usage: quintic_slope_one(torch.tensor(0.25))  # 0.5
    """
    return 1.0 - 2.0 * r1


def quintic_order(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """Order of convergence u_k -> 1 (1, 2 or 3, as an int64 tensor) for an
    admissible quintic: 1 if r1 < 1/2, 2 if r1 = 1/2 and r2 != 3/8, and 3 for
    (r1, r2) = (1/2, 3/8). The cases are selected by exact comparison.

    Usage: quintic_order(torch.tensor(0.5), torch.tensor(0.25))  # 2
    """
    r1, r2 = torch.broadcast_tensors(r1, r2)
    is_p2 = (r1 == 0.5) & (r2 == 0.375)
    # 2 by default, 1 below the edge r1 = 1/2, 3 at the truncated quintic
    return 2 - (r1 < 0.5).to(torch.int64) + is_p2.to(torch.int64)


def quintic_error_constant(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """Asymptotic error constant of `quintic_order`: 1 - 2 r1 for order 1,
    3/2 - 4 r2 for order 2 and 5/2 for order 3.

    Usage: quintic_error_constant(torch.tensor(0.5), torch.tensor(0.25))  # 0.5
    """
    r1, r2 = torch.broadcast_tensors(r1, r2)
    order = quintic_order(r1, r2)
    return torch.where(
        order == 1,
        1.0 - 2.0 * r1,
        torch.where(order == 2, 1.5 - 4.0 * r2, torch.full_like(r1, 2.5)),
    )
