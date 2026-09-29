"""Matrix sign map, two ways: exact (through an SVD) or approximate (Newton-Schulz
steps, no SVD needed).

Everything that computes a sign is a function matrix -> matrix (`msgn`), so the
exact and approximate versions can be swapped freely, e.g. inside `cpwl.py`.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from ns_core import metrics, ns_iteration

msgn = Callable[[torch.Tensor], torch.Tensor]


def sgn_exact(t: torch.Tensor) -> torch.Tensor:
    """
    t: input tensor
    Returns: elementwise sign, 0 stays 0
    """
    return torch.sign(t)


def sgn_svd(M: torch.Tensor, tol: float | None = None, driver: str | None = None) -> torch.Tensor:
    """
    M: input matrix
    tol: zero cutoff (default: numerical rank)
    driver: CUDA SVD driver
    Returns: exact sign matrix
    """
    U, sigma, V = metrics.reference_svd(M, driver=driver)
    if tol is None:
        tol = metrics.numerical_rank_tol(M, sigma)
    signs = (sigma > tol).to(M.dtype)
    return (U * signs) @ V.mH


def spectral_norm_exact(M: torch.Tensor) -> torch.Tensor:
    """
    M: input matrix
    Returns: largest singular value
    """
    return torch.linalg.matrix_norm(M, ord=2)


def spectral_norm_power(
    M: torch.Tensor, iters: int, *, generator: torch.Generator
) -> torch.Tensor:
    """
    M: input matrix
    iters: power steps
    generator: RNG
    Returns: largest singular value estimate
    Note: never overshoots, so callers add a margin
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
    """
    M: input matrix
    D: degree
    K: steps
    power_iters: norm-estimate steps
    margin: norm safety factor
    generator: RNG
    Returns: approximate sign matrix
    Note: fixed K, no checks; zero matrix not handled
    """
    s = margin * spectral_norm_power(M, power_iters, generator=generator)
    X = M / s
    coeffs = ns_iteration.bpoly_coeffs(D, dtype=M.dtype, device=M.device)
    for _ in range(K):
        X = ns_iteration.ns_step_gram(X, coeffs)
    return X


def sgn_ns_until(
    M: torch.Tensor,
    D: int,
    eps: float,
    k_max: int,
    *,
    power_iters: int,
    margin: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int, bool]:
    """
    M: input matrix (full rank)
    D: degree
    eps: target residual
    k_max: step cap
    power_iters: norm-estimate steps
    margin: norm safety factor
    generator: RNG
    Returns: X, steps K, reached target?
    Note: one host sync per step
    """
    s = margin * spectral_norm_power(M, power_iters, generator=generator)
    X = M / s
    coeffs = ns_iteration.bpoly_coeffs(D, dtype=M.dtype, device=M.device)
    r = min(M.shape[-2:])
    eye = torch.eye(r, dtype=M.dtype, device=M.device)
    level = eps * math.sqrt(r)
    for K in range(k_max + 1):
        G = ns_iteration.gram_matrix(X)
        if bool(torch.linalg.norm(G - eye) <= level):
            return X, K, True
        if K < k_max:
            X = ns_iteration.ns_step_gram(X, coeffs, gram=G)
    return X, k_max, False


def make_sgn_ns_until(
    D: int,
    eps: float,
    k_max: int,
    *,
    power_iters: int,
    margin: float,
    generator: torch.Generator,
    stats: list[tuple[int, bool]] | None = None,
) -> msgn:
    """
    D: degree
    eps: target residual
    k_max: step cap
    power_iters: norm-estimate steps
    margin: norm safety factor
    generator: RNG
    stats: list to log (K, reached) per call
    Returns: msgn function (matrix -> sign)
    """

    def msgn(M: torch.Tensor) -> torch.Tensor:
        X, K, reached = sgn_ns_until(
            M, D, eps, k_max, power_iters=power_iters, margin=margin, generator=generator
        )
        if stats is not None:
            stats.append((K, reached))
        return X

    return msgn


def make_sgn_ns(
    D: int,
    n_iters: int,
    *,
    scale: torch.Tensor | float | None = None,
    tol: float | None = None,
) -> msgn:
    """
    D: degree
    n_iters: steps
    scale: divisor for M (default: spectral norm, recomputed per call)
    tol: early stop when a step changes less than this
    Returns: msgn function (matrix -> sign)
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
