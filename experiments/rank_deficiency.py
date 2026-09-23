"""Rank-deficiency and conditioning sweeps for the decomposition-free matrix
sign map (reviewer comments: "the rank-deficient experiment deserves more
attention ... experiments with several ranks and several smallest nonzero
singular values would provide much stronger evidence", and "a plot of the
iteration count or error decay for several values of sigma_min would
directly validate the theoretical analysis").

Two independent sweeps, both reported as iterations-to-a-fixed-tolerance
against the swept parameter, one curve per degree D:

- `rank_sweep` varies the number of exact-zero singular values `n_zero`
  (the rank deficiency itself), at a fixed matrix size, leaving the nonzero
  part of the spectrum at whatever a Gaussian draw gives it. This is the
  qualitative question: does the decomposition-free iteration stay
  meaningful as the matrix gets more degenerate.
- `conditioning_sweep` varies the smallest nonzero singular value sigma_min
  itself (via `n_small=1`), at fixed rank. This is the quantitative
  question Chapter 4 predicts an answer to: convergence slows as sigma_min
  approaches zero, at the rate the scalar orbit u_{k+1} = p_D(u_k) started
  at u0 = sigma_min already predicts (cor:matrix-order, exercised the same
  way in experiments.convergence_order).

Each swept value currently uses a single random matrix (no repeats/averaging
yet - see experiments/scripts/rank_deficiency.py; that's the obvious next
step once the shape of these sweeps is settled).

Both sweeps use the exact spectral-norm error ||X_k - Sgn(M)||_2 as the
error norm, not the relative Frobenius error used elsewhere in the project
(the reviewer's comment on stating the norm explicitly applies here too):
this is the quantity cor:matrix-order identifies with the scalar orbit's
error exactly, at every step, so it is what lets a matrix-level curve be
checked directly against the scalar theory. It also has a clean floor
(exactly 0, not just small) on the zero singular values of a rank-deficient
matrix, since sgn(0) = 0 is already the fixed point p_D maps 0 to.
"""

from __future__ import annotations

import torch

from ns_core import matrices, metrics, ns_iteration, sign_map


def _error_orbit(
    m: int,
    n: int,
    D: int,
    n_iters: int,
    *,
    n_zero: int,
    n_small: int,
    s_small: float,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Spectral-norm error orbit ||X_k - Sgn(M)||_2, k = 0..n_iters, for one
    random rank-deficient/ill-conditioned M (see matrices.rand_rank_deficient
    for what n_zero/n_small/s_small control)."""
    M = matrices.rand_rank_deficient(
        m, n, generator=generator, n_zero=n_zero, n_small=n_small, s_small=s_small,
        device=device, dtype=dtype,
    )
    target = sign_map.sgn_svd(M)
    orbit = ns_iteration.ns_orbit_matrix(M, D, n_iters)
    tiny = torch.finfo(dtype).tiny
    return torch.stack([torch.clamp(metrics.spectral_error(X, target), min=tiny) for X in orbit])


def _iterations_to_tolerance(errors: torch.Tensor, tol: float) -> int:
    """First index k with errors[k] <= tol, or -1 if the orbit never reaches
    it within the given budget (a signal n_iters was too small for this
    point, reported as such rather than silently clipped)."""
    below = (errors <= tol).nonzero(as_tuple=True)[0]
    return int(below[0]) if below.numel() else -1


def rank_sweep(
    n_zero_values: list[int],
    m: int,
    n: int,
    D_values: list[int],
    n_iters: int,
    *,
    tol: float = 1e-10,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> dict[int, dict[int, int]]:
    """Iterations-to-tolerance against rank deficiency, one series per D.

    For each D in D_values and each n_zero in n_zero_values, draws a single
    random m x n matrix with exactly n_zero zero singular values
    (`matrices.rand_rank_deficient`, n_small=0 so the nonzero part of the
    spectrum is left at its Gaussian draw) and records how many NS steps of
    degree D it takes to bring the spectral-norm error to `tol` (-1 if it
    doesn't, within n_iters).

    Returns {D: {n_zero: iterations}}. Requires n_zero < min(m, n) for
    every value (a fully zero matrix is a degenerate edge case with an
    all-zero target, excluded rather than silently included).
    """
    r = min(m, n)
    if any(nz < 0 or nz >= r for nz in n_zero_values):
        raise ValueError(f"n_zero values must lie in [0, {r - 1}]")
    return {
        D: {
            n_zero: _iterations_to_tolerance(
                _error_orbit(
                    m, n, D, n_iters, n_zero=n_zero, n_small=0, s_small=0.0,
                    generator=generator, dtype=dtype, device=device,
                ),
                tol,
            )
            for n_zero in n_zero_values
        }
        for D in D_values
    }


def conditioning_sweep(
    sigma_min_values: list[float],
    m: int,
    n: int,
    D_values: list[int],
    n_iters: int,
    *,
    tol: float = 1e-10,
    generator: torch.Generator,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> dict[int, dict[float, int]]:
    """Iterations-to-tolerance against the smallest nonzero singular value,
    one series per D.

    For each D in D_values and each sigma_min in sigma_min_values, draws a
    single random full-rank m x n matrix whose smallest singular value is
    exactly sigma_min (`matrices.rand_rank_deficient`, n_small=1,
    s_small=sigma_min) and records how many NS steps of degree D it takes
    to bring the spectral-norm error to `tol` (-1 if it doesn't, within
    n_iters).

    Returns {D: {sigma_min: iterations}}. Every sigma_min should be well
    below 1 for `matrices.rand_rank_deficient` to guarantee it ends up the
    true minimum of the spectrum (see that function's docstring); this is
    the near-zero regime it is built for.
    """
    return {
        D: {
            sigma_min: _iterations_to_tolerance(
                _error_orbit(
                    m, n, D, n_iters, n_zero=0, n_small=1, s_small=sigma_min,
                    generator=generator, dtype=dtype, device=device,
                ),
                tol,
            )
            for sigma_min in sigma_min_values
        }
        for D in D_values
    }