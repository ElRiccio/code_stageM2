"""The matrix sign map msgn: the exact SVD-based version and the
decomposition-free surrogate obtained by running `ns_iteration` on a
rescaled matrix.

`msgn` is the type of a one-argument callable Tensor -> Tensor computing a
matrix sign; `sgn_svd(M)` and `make_sgn_ns(D, n_iters)` both produce
callables of this type, so either can be passed wherever a matrix sign map
is expected.
"""

from __future__ import annotations

from typing import Callable

import torch

from ns_core import metrics, ns_iteration

msgn = Callable[[torch.Tensor], torch.Tensor]


def sgn_exact(t: torch.Tensor) -> torch.Tensor:
    """Scalar sign with sgn(0) = 0, applied elementwise.

    Usage: sgn_exact(torch.tensor([-2.0, 0.0, 3.0]))
    """
    return torch.sign(t)


def sgn_svd(M: torch.Tensor, tol: float | None = None, driver: str | None = None) -> torch.Tensor:
    """Exact msgn(M) = U diag(sgn(sigma)) V^T from a thin SVD. Singular
    values at or below `tol` count as zero; `tol` defaults to
    `metrics.numerical_rank_tol(M, sigma)`. `driver` is the CUDA SVD driver
    (see `metrics.reference_svd`).

    Usage: N = sgn_svd(M)
    """
    U, sigma, V = metrics.reference_svd(M, driver=driver)
    if tol is None:
        tol = metrics.numerical_rank_tol(M, sigma)
    signs = (sigma > tol).to(M.dtype)
    return (U * signs) @ V.mH


def spectral_norm_exact(M: torch.Tensor) -> torch.Tensor:
    """Largest singular value of M via torch.linalg, the default rescaling
    constant: msgn(M / beta) = msgn(M) for every beta > 0.

    Usage: beta = spectral_norm_exact(M)
    """
    return torch.linalg.matrix_norm(M, ord=2)


def spectral_norm_power(
    M: torch.Tensor, iters: int, *, generator: torch.Generator
) -> torch.Tensor:
    """Estimate of the largest singular value of M by `iters` power steps on
    M^T M from a random start. The estimate never exceeds the true value, so
    callers multiply it by a margin. Uses matrix-vector products only and no
    host synchronization.

    Usage: beta = spectral_norm_power(M, 10, generator=g)
    """
    v = torch.randn(M.shape[-1], generator=generator, device=M.device, dtype=M.dtype)
    v = v / torch.linalg.norm(v)
    for _ in range(iters):
        v = M.mH @ (M @ v)
        v = v / torch.linalg.norm(v)
    return torch.linalg.norm(M @ v)


def sgn_ns_fixed(
    M: torch.Tensor,
    D: int,
    K: int,
    *,
    power_iters: int,
    margin: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Decomposition-free msgn(M): K steps of degree D (min-side Gram form,
    `ns_iteration.ns_step_gram`) on M / (margin * power estimate of sigma_max).
    There is no residual check, no zero-scale test and no stored orbit, so
    nothing forces a device synchronization; a zero matrix is not handled.

    Usage: X = sgn_ns_fixed(M, D=2, K=8, power_iters=10, margin=1.1, generator=g)
    """
    s = margin * spectral_norm_power(M, power_iters, generator=generator)
    X = M / s
    coeffs = ns_iteration.bpoly_coeffs(D, dtype=M.dtype, device=M.device)
    for _ in range(K):
        X = ns_iteration.ns_step_gram(X, coeffs)
    return X


def make_sgn_ns(
    D: int,
    n_iters: int,
    *,
    scale: torch.Tensor | float | None = None,
    tol: float | None = None,
) -> msgn:
    """The surrogate msgn: a callable running `n_iters` steps of degree D on
    M / scale, with `scale` defaulting to the spectral norm of M. The scale
    is recomputed at every call, so the callable applies to any argument
    (for instance alpha * N - M). With `tol`, iteration stops once the step
    ||X_{k+1} - X_k||_F falls below it.

    Usage: msgn_ns = make_sgn_ns(D=2, n_iters=8); X = msgn_ns(M)
    """

    def msgn(M: torch.Tensor) -> torch.Tensor:
        s = spectral_norm_exact(M) if scale is None else scale
        s = torch.as_tensor(s, dtype=M.dtype, device=M.device)
        if s <= 0:
            return torch.zeros_like(M)
        coeffs = ns_iteration.bpoly_coeffs(D, dtype=M.dtype, device=M.device)
        X = M / s
        for _ in range(n_iters):
            X_prev = X
            X = ns_iteration.ns_step_matrix(X, coeffs)
            if tol is not None and torch.linalg.norm(X - X_prev) < tol:
                break
        return X

    return msgn
