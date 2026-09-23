"""Numerical validation of the exact order of convergence (reviewer comment:
"the predicted order D+1 could be checked numerically by plotting suitable
error ratios or by estimating the slope on a log-log plot").

Two complementary checks, both estimating the order from three consecutive
log-errors L_k = log10(e_k) via the standard formula

    p_hat_k = (L_{k+2} - L_{k+1}) / (L_{k+1} - L_k),

which is p in e_{k+1} ~ C e_k^p read off without needing C:

- `scalar_order_validation` / `degree_sweep_scalar` estimate the order
  directly from the scalar orbit u_{k+1} = p_D(u_k), using
  ns_core.ns_iteration.log10_error_orbit's deflated evaluation to keep going
  arbitrarily far past the point where the raw error 1 - u_k underflows.
  This is where the exact rate D+1 of prop:order is seen cleanly, since the
  deflated recursion has no floating-point floor of its own.
- `matrix_order_validation` / `degree_sweep_matrix` run the actual
  decomposition-free matrix iteration (ns_core.ns_iteration.ns_orbit_matrix)
  on a matrix with a prescribed smallest nonzero singular value sigma_min,
  and compare its spectral-norm error against the scalar orbit started at
  u0 = sigma_min, which cor:matrix-order identifies it with exactly (bpoly_D
  is increasing on [0, 1], so the index attaining the max error never
  changes). Unlike the scalar check, this one is bounded by machine
  precision: once the matrix error underflows towards the dtype's rounding
  floor, further floating-point subtraction is noise and the empirical
  order estimate stops being meaningful. That ceiling is reported alongside
  the estimate rather than hidden — it is a real fact about the
  decomposition-free algorithm, not a bug in the experiment, and it is
  exactly why the scalar check above is done with the deflated form instead
  of by direct subtraction.
"""

from __future__ import annotations

import math

import torch

from ns_core import matrices, metrics, ns_iteration, sign_map


def _empirical_order(log10_errors: torch.Tensor) -> torch.Tensor:
    """p_hat[k] = (L[k+2] - L[k+1]) / (L[k+1] - L[k]) for k = 0..len-3, the
    three-point estimate of the order p in e_{k+1} ~ C e_k^p, applied
    directly to log10 errors L_k = log10 e_k. Length len(log10_errors) - 2.
    """
    L = log10_errors
    if L.numel() < 3:
        return torch.empty(0, dtype=L.dtype, device=L.device)
    return (L[2:] - L[1:-1]) / (L[1:-1] - L[:-2])


# ----------------------------------------------------------------------------
# Scalar orbit: the exact check, via the deflated evaluation
# ----------------------------------------------------------------------------


def scalar_order_validation(
    D: int, u0: float, n_iters: int, *, switch: float = 1e-8
) -> dict[str, torch.Tensor | float | int]:
    """Empirical order of the scalar orbit u_{k+1} = p_D(u_k) started at u0.

    prop:order predicts p_hat_k -> D+1, with the ratio e_{k+1}/(1-u_k)^(D+1)
    itself converging to the exact asymptotic constant kappa_D (both
    returned alongside the estimate, for comparison).
    """
    log10_err = ns_iteration.log10_error_orbit(u0, D, n_iters, switch=switch)
    return {
        "log10_error": log10_err,
        "order_estimate": _empirical_order(log10_err),
        "predicted_order": D + 1,
        "asymptotic_constant": ns_iteration.asymptotic_error_constant(D),
        "D": D,
        "u0": u0,
    }


def degree_sweep_scalar(
    D_values: list[int], u0: float, n_iters: int, *, switch: float = 1e-8
) -> dict[int, dict]:
    """scalar_order_validation for each D in D_values, at the same u0."""
    return {D: scalar_order_validation(D, u0, n_iters, switch=switch) for D in D_values}


# ----------------------------------------------------------------------------
# Matrix iteration: the decomposition-free check, bounded by precision
# ----------------------------------------------------------------------------


def matrix_order_validation(
    D: int,
    sigma_min: float,
    n: int,
    n_iters: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor | float | int]:
    """Empirical order of the matrix NS iteration on a random n x n matrix
    with spectral norm 1 and smallest nonzero singular value sigma_min.

    The spectrum is prescribed exactly (linspace(1, sigma_min, n)) via
    matrices.rand_prescribed_spectrum rather than by overwriting the tail of
    a Gaussian matrix's spectrum (rand_rank_deficient): sigma_min here is
    meant to be comparable to the rest of the spectrum, not near-zero, and
    only the direct construction guarantees it ends up the true minimum.

    By cor:matrix-order, ||X_k - Sgn(M)||_2 coincides with the scalar
    orbit's error at u0 = sigma_min for every k; `matrix_log10_error` and
    `scalar_log10_error` should agree until the matrix computation reaches
    its floating-point floor (reported as `precision_floor`, log10 of the
    dtype's machine epsilon), which the scalar side does not have.

    Defaults to float64 to push that floor as low as this iteration can go;
    pass dtype=torch.float32 to see the (much shallower) floor relevant to
    the mixed-precision/GPU timing experiments.
    """
    sigma = torch.linspace(1.0, sigma_min, n, dtype=dtype)
    M = matrices.rand_prescribed_spectrum(
        n, n, sigma, generator=generator, device=device, dtype=dtype
    )
    target = sign_map.sgn_svd(M)
    orbit = ns_iteration.ns_orbit_matrix(M, D, n_iters)

    tiny = torch.finfo(dtype).tiny
    errors = torch.stack(
        [torch.clamp(metrics.spectral_error(X, target), min=tiny) for X in orbit]
    )
    matrix_log10_err = torch.log10(errors.to(torch.float64))
    scalar_log10_err = ns_iteration.log10_error_orbit(sigma_min, D, n_iters)

    return {
        "matrix_log10_error": matrix_log10_err,
        "scalar_log10_error": scalar_log10_err,
        "order_estimate": _empirical_order(matrix_log10_err),
        "predicted_order": D + 1,
        "D": D,
        "sigma_min": sigma_min,
        # log10 of the dtype's machine epsilon: the noise floor below which
        # this matrix-level error (computed by direct subtraction) is no
        # longer meaningful.
        "precision_floor": math.log10(torch.finfo(dtype).eps),
    }


def degree_sweep_matrix(
    D_values: list[int],
    sigma_min: float,
    n: int,
    n_iters: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> dict[int, dict]:
    """matrix_order_validation for each D in D_values, same sigma_min/n."""
    return {
        D: matrix_order_validation(
            D, sigma_min, n, n_iters, generator=generator, dtype=dtype, device=device
        )
        for D in D_values
    }
