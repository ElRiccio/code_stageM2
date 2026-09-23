"""The matrix sign map: the exact reference and the decomposition-free
surrogate built by running ns_core.ns_iteration on a pre-scaled matrix.

Every decomposition-free spectral operator in ns_core.cpwl is expressed in
terms of a `Sgn` callable, so that swapping in the exact reference
(sgn_svd) or a truncated surrogate (make_sgn_ns with different D, n_iters)
is a one-line change at the call site, never a change to the operator's own
formula.
"""

from __future__ import annotations

from typing import Callable

import torch

from ns_core import metrics, ns_iteration

Sgn = Callable[[torch.Tensor], torch.Tensor]


def sgn_exact(t: torch.Tensor) -> torch.Tensor:
    """Scalar sign, convention sgn(0) = 0."""
    return torch.sign(t)


def sgn_svd(M: torch.Tensor, tol: float | None = None) -> torch.Tensor:
    """Exact matrix sign U diag(sign(sigma)) V^T, thresholded at `tol`
    (defaults to ns_core.metrics.numerical_rank_tol(M, sigma)).

    This is the reference every decomposition-free evaluation in ns_core.cpwl
    is checked against; it is deliberately not the fast path (it costs a
    full SVD), and should never be called from inside a "decomposition-free"
    code path.
    """
    U, sigma, V = metrics.reference_svd(M)
    if tol is None:
        tol = metrics.numerical_rank_tol(M, sigma)
    signs = (sigma > tol).to(M.dtype)
    return (U * signs) @ V.mH


def spectral_norm_exact(M: torch.Tensor) -> torch.Tensor:
    """Exact largest singular value of M, via torch.linalg.

    The pre-scaling norm used by default: Sgn(M / beta) = Sgn(M) for any
    beta > 0, so any positive scale that puts the spectrum of M / beta into
    (-1, 1) leaves the target unchanged. Computing it exactly (rather than by
    power iteration) is itself a decomposition-based step; a matrix-free
    alternative belongs here as a second pre-scaling option once the timing
    comparison (experiments.svd_comparison) needs one.
    """
    return torch.linalg.matrix_norm(M, ord=2)


def make_sgn_ns(
    D: int,
    n_iters: int,
    *,
    scale: torch.Tensor | float | None = None,
    tol: float | None = None,
) -> Sgn:
    """Sgn surrogate as a one-argument callable: `n_iters` NS steps of degree
    D, on M pre-scaled to unit spectral norm (or to `scale`, if given).

    Every call pre-scales its own argument. This matters wherever a Sgn
    callable is invoked on a matrix shifted away from M (e.g. the clip and
    soft-threshold sign forms in ns_core.cpwl evaluate Sgn on `alpha * N -
    M`, not on M itself), so the surrogate must not assume its argument is
    already unit-norm.
    """

    def Sgn(M: torch.Tensor) -> torch.Tensor:
        s = spectral_norm_exact(M) if scale is None else scale
        s = torch.as_tensor(s, dtype=M.dtype, device=M.device)
        if s <= 0:
            return torch.zeros_like(M)
        coeffs = ns_iteration.bpoly_coeffs(D, dtype=M.dtype).to(device=M.device)
        X = M / s
        for _ in range(n_iters):
            X_prev = X
            X = ns_iteration.ns_step_matrix(X, coeffs)
            if tol is not None and torch.linalg.norm(X - X_prev) < tol:
                break
        return X

    return Sgn
