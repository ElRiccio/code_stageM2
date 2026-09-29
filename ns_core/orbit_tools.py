"""Shared helpers for the convergence experiments: build a test matrix with a
known spectrum, run the iteration on it, and measure the errors.

The test matrix and its exact sign are built in float64. The iteration runs in
the requested dtype and errors are measured back in float64, so any accuracy
floor in a curve comes from the iteration, not the reference.
"""

from __future__ import annotations

import math
import warnings

import torch

from ns_core import matrices, metrics, ns_iteration, sign_map


def resolve_rank(m: int, n: int, rank: int | None) -> int:
    """
    m, n: shape
    rank: None = full, <= 0 counts down from full, > 0 kept
    Returns: actual rank
    """
    r = min(m, n)
    if rank is None:
        return r
    if rank <= 0:
        rank += r
    if not 1 <= rank <= r:
        raise ValueError("rank must lie in [1, min(m, n)] once resolved")
    return rank


def log_spectrum(
    r: int,
    smin: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    r: how many values
    smin: smallest value
    dtype, device: output type and place
    Returns: log-spaced values from 1 down to smin
    """
    return torch.logspace(0.0, math.log10(smin), r, dtype=dtype, device=device)


def make_instance(
    m: int,
    n: int,
    rank: int | None,
    smin: float,
    seed: int,
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    m, n: shape
    rank: see resolve_rank
    smin: smallest nonzero singular value
    seed: RNG seed
    device: where to build it
    Returns: matrix M, its exact sign N
    """
    r = resolve_rank(m, n, rank)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    sigma = log_spectrum(r, smin, device=device)
    M = matrices.rand_prescribed_spectrum(
        m, n, sigma, generator=g, device=device, dtype=torch.float64
    )
    return M, sign_map.sgn_svd(M)


def default_eps(dtype: torch.dtype) -> float:
    """
    dtype: working precision
    Returns: default freeze precision (10 machine epsilons)
    """
    return 10.0 * torch.finfo(dtype).eps


def run_orbit(
    M: torch.Tensor,
    N: torch.Tensor,
    D: int,
    k_max: int,
    dtype: torch.dtype,
    eps: float | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    M: input matrix (largest singular value 1)
    N: exact sign of M
    D: degree
    k_max: steps
    dtype: precision to run in
    eps: freeze precision (None = default_eps)
    Returns: list of k_max + 1 iterates in float64, spectral-norm error per iterate
    Note: once the error is <= eps the iterate and its error are held to k_max, so
    rounding noise in the null space is never amplified; below default_eps it never fires
    """
    if eps is None:
        eps = default_eps(dtype)
    elif eps < default_eps(dtype):
        warnings.warn(f"eps={eps:g} is below the {dtype} floor, so the freeze may never fire")
    coeffs = ns_iteration.bpoly_coeffs(D, dtype=dtype, device=M.device)
    X = M.to(dtype)
    orbit, errors = [], []
    frozen = False
    for k in range(k_max + 1):
        if not frozen:
            X64 = X.to(torch.float64)
            err = metrics.spectral_error(X64, N)
            frozen = bool(err <= eps)
        orbit.append(X64)
        errors.append(err)
        if not frozen and k < k_max:
            X = ns_iteration.ns_step_matrix(X, coeffs)
    return orbit, torch.stack(errors)


def orbit_ranks(orbit: list[torch.Tensor], tol: float) -> torch.Tensor:
    """
    orbit: iterates
    tol: cutoff relative to largest singular value
    Returns: numerical rank per iterate
    """
    sv = torch.stack([torch.linalg.svdvals(X) for X in orbit])
    return (sv > tol * sv[:, :1]).sum(dim=1)


def first_hit(err: torch.Tensor, eps: float) -> int | None:
    """
    err: error curve
    eps: target
    Returns: first index at or below eps, None if never
    """
    hits = (err <= eps).nonzero()
    return int(hits[0]) if hits.numel() else None


def report_iterations(
    iterations: dict[int, int | None], predicted: dict[int, int] | None = None
) -> None:
    """
    iterations: first hit per degree (res["iterations"])
    predicted: K_D per degree, shown alongside
    Returns: nothing, prints one line per degree
    """
    for D in sorted(iterations):
        k = iterations[D]
        line = f"D={D}: " + ("not reached within k_max" if k is None else f"{k} iterations")
        if predicted is not None and D in predicted:
            line += f"  (predicted K_D = {predicted[D]})"
        print(line)
