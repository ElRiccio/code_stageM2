"""Helpers shared by the convergence experiments: the test instance with an
exactly prescribed spectrum, the orbit of the iteration under test, and its
measurements against msgn(M).

The instance is built (and its reference sign computed with an SVD) in
float64 on the test side; the iteration itself runs in the requested dtype
and uses matrix products only. Measurements are taken in float64, so the
accuracy floor seen in a curve is the iteration's, not the reference's.
"""

from __future__ import annotations

import math

import torch

from ns_core import matrices, metrics, ns_iteration, sign_map


def resolve_rank(m: int, n: int, rank: int | None) -> int:
    """Rank of the test matrix: None gives min(m, n), a value <= 0 counts
    down from min(m, n) (-1 is min(m, n) - 1), a positive value is kept.

    Usage: resolve_rank(128, 64, -1)  # 63
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
    """r singular values log-spaced from 1 down to smin (just [1] if r = 1).

    Usage: sigma = log_spectrum(32, 1e-3)
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
    """(M, N): an m x n float64 matrix of the given rank whose nonzero
    singular values are log-spaced in [smin, 1], and N = msgn(M) from the
    exact SVD. `rank` follows `resolve_rank`.

    Usage: M, N = make_instance(128, 128, None, 1e-2, seed=0)
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
    """Default stopping tolerance on the step ||X_{k+1} - X_k||_F: the square
    root of the machine epsilon of `dtype`. With convergence of order two or
    more, the iterate after such a step is already at rounding level, so
    stopping there keeps the null space from amplifying rounding noise.

    Usage: tol = default_tol(torch.float64)
    """
    return math.sqrt(torch.finfo(dtype).eps)


def run_orbit(
    M: torch.Tensor,
    D: int,
    k_max: int,
    dtype: torch.dtype,
    tol: float | None = None,
) -> list[torch.Tensor]:
    """Iterates X_0, ..., X_{k_max} of degree D started from M (whose largest
    singular value is 1, so no rescaling), run in `dtype` and returned in
    float64. With `tol`, iteration stops once ||X_{k+1} - X_k||_F < tol and
    the last iterate is repeated.

    Usage: orbit = run_orbit(M, D=2, k_max=20, dtype=torch.float32, tol=1e-4)
    """
    orbit = ns_iteration.ns_orbit_matrix(M.to(dtype), D, k_max, scale=1.0, tol=tol)
    return [X.to(torch.float64) for X in orbit]


def orbit_errors(orbit: list[torch.Tensor], N: torch.Tensor) -> torch.Tensor:
    """Spectral-norm errors ||X_k - N||_2 along an orbit, shape (K + 1,).

    Usage: e = orbit_errors(orbit, N)
    """
    return torch.stack([metrics.spectral_error(X, N) for X in orbit])


def orbit_ranks(orbit: list[torch.Tensor], tol: float) -> torch.Tensor:
    """Numerical rank of every iterate, shape (K + 1,): the number of singular
    values above tol * sigma_max.

    Usage: ranks = orbit_ranks(orbit, tol=1e-8)
    """
    sv = torch.stack([torch.linalg.svdvals(X) for X in orbit])
    return (sv > tol * sv[:, :1]).sum(dim=1)


def first_hit(err: torch.Tensor, eps: float) -> int | None:
    """First index k with err[k] <= eps, or None if the curve never gets there.

    Usage: k = first_hit(e, 1e-6)
    """
    hits = (err <= eps).nonzero()
    return int(hits[0]) if hits.numel() else None


def report_iterations(
    iterations: dict[int, int | None], predicted: dict[int, int] | None = None
) -> None:
    """Prints, for each D, the first iteration at which the error is at or
    below eps (as returned in res["iterations"]), with the predicted K_D
    alongside if given.

    Usage: report_iterations(res["iterations"], res.get("predicted"))
    """
    for D in sorted(iterations):
        k = iterations[D]
        line = f"D={D}: " + ("not reached within k_max" if k is None else f"{k} iterations")
        if predicted is not None and D in predicted:
            line += f"  (predicted K_D = {predicted[D]})"
        print(line)
