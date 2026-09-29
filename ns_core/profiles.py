"""Closed-form quantities for the iteration polynomials: the truncated series
family (coefficients, basin radius, iteration count) and the admissible
quintics (coefficients, slopes, order and constant of convergence).

Coefficient tensors use the same layout as `ns_iteration.bpoly_coeffs`, so
`ns_iteration.ns_step_matrix` accepts them directly.
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
    """
    D: degree
    dtype, device: output type and place
    Returns: series coefficients, ascending
    """
    if D < 0:
        raise ValueError("D must be nonnegative")
    coeffs = [ns_iteration.central_binomial(j) / 4.0**j for j in range(D + 1)]
    return torch.tensor(coeffs, dtype=dtype, device=device)


def truncated_series_eval(t: torch.Tensor, D: int) -> torch.Tensor:
    """
    t: points
    D: degree
    Returns: truncated series value, same shape as t
    """
    coeffs = truncated_series_coeffs(D, dtype=t.dtype, device=t.device)
    out = torch.zeros_like(t)
    for c in reversed(coeffs):
        out = out * t + c
    return out


def basin_radius(D: int) -> tuple[float, float]:
    """
    D: degree (>= 1)
    Returns: basin radius R_D, root t_D behind it
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
    """
    D: degree
    theta: starting residual in (0, 1)
    eps: target accuracy in (0, 1)
    Returns: K_D, steps that guarantee the target
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
    """
    r1, r2: quintic parameters (broadcast together)
    Returns: coefficients, shape (..., 3)
    """
    r1, r2 = torch.broadcast_tensors(r1, r2)
    return torch.stack([1.0 + r1 + r2, -(r1 + 2.0 * r2), r2], dim=-1)


def quintic_slope_origin(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """
    r1, r2: quintic parameters
    Returns: slope at 0
    """
    return 1.0 + r1 + r2


def quintic_slope_one(r1: torch.Tensor) -> torch.Tensor:
    """
    r1: quintic parameter
    Returns: slope at 1
    """
    return 1.0 - 2.0 * r1


def quintic_order(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """
    r1, r2: quintic parameters
    Returns: convergence order (1, 2 or 3), int64
    Note: exact float comparison picks the case
    """
    r1, r2 = torch.broadcast_tensors(r1, r2)
    is_p2 = (r1 == 0.5) & (r2 == 0.375)
    # 2 by default, 1 below the edge r1 = 1/2, 3 at the truncated quintic
    return 2 - (r1 < 0.5).to(torch.int64) + is_p2.to(torch.int64)


def quintic_error_constant(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """
    r1, r2: quintic parameters
    Returns: error constant for that convergence order
    """
    r1, r2 = torch.broadcast_tensors(r1, r2)
    order = quintic_order(r1, r2)
    return torch.where(
        order == 1,
        1.0 - 2.0 * r1,
        torch.where(order == 2, 1.5 - 4.0 * r2, torch.full_like(r1, 2.5)),
    )
