"""Shared helpers for the convergence experiments: build a test matrix with a
known spectrum, run the iteration on it, and measure the errors.

The test matrix and its exact sign are built in float64. The iteration runs in
the requested dtype and errors are measured back in float64, so any accuracy
floor in a curve comes from the iteration, not the reference.
"""

from __future__ import annotations

import math

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


def default_tol(dtype: torch.dtype) -> float:
    """
    dtype: working precision
    Returns: default early-stop tolerance (sqrt of machine epsilon)
    """
    return math.sqrt(torch.finfo(dtype).eps)


def run_orbit(
    M: torch.Tensor,
    D: int,
    k_max: int,
    dtype: torch.dtype,
    tol: float | None = None,
) -> list[torch.Tensor]:
    """
    M: input matrix (largest singular value 1)
    D: degree
    k_max: steps
    dtype: precision to run in
    tol: early stop when a step changes less than this
    Returns: list of k_max + 1 iterates, in float64
    """
    orbit = ns_iteration.ns_orbit_matrix(M.to(dtype), D, k_max, scale=1.0, tol=tol)
    return [X.to(torch.float64) for X in orbit]


def orbit_errors(orbit: list[torch.Tensor], N: torch.Tensor) -> torch.Tensor:
    """
    orbit: iterates
    N: exact sign
    Returns: spectral-norm error per iterate
    """
    return torch.stack([metrics.spectral_error(X, N) for X in orbit])


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
