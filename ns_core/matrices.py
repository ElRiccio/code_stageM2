"""Random test matrices: Gaussian, symmetric, prescribed spectrum, rank-deficient
or ill-conditioned.

Every function takes a torch.Generator (on the same device) plus device and
dtype keywords, defaulting to CPU and float32.
"""

from __future__ import annotations

import torch


def rand_gaussian(
    m: int,
    n: int,
    *,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    m, n: shape
    generator: RNG
    device, dtype: where and how to store
    Returns: matrix of standard normal entries
    """
    return torch.randn(m, n, generator=generator, device=device, dtype=dtype)


def rand_orthogonal_factor(
    q: int,
    k: int,
    *,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    q, k: shape (q >= k)
    generator: RNG
    device, dtype: where and how to store
    Returns: q x k matrix with orthonormal columns
    """
    if k > q:
        raise ValueError("need q >= k for a semi-orthogonal q x k factor")
    A = rand_gaussian(q, k, generator=generator, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    s = torch.sign(torch.diagonal(R))
    s = torch.where(s == 0, torch.ones_like(s), s)  # pin the sign, so runs reproduce
    return Q * s


def rand_symmetric(
    n: int,
    *,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    normalize: bool = True,
) -> torch.Tensor:
    """
    n: size
    generator: RNG
    device, dtype: where and how to store
    normalize: scale to spectral norm 1
    Returns: random symmetric matrix
    """
    A = rand_gaussian(n, n, generator=generator, device=device, dtype=dtype)
    S = 0.5 * (A + A.T)
    if normalize:
        beta = torch.linalg.svdvals(S)[0]
        if beta > 0:
            S = S / beta
    return S


def rand_prescribed_spectrum(
    m: int,
    n: int,
    sigma: torch.Tensor,
    *,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    m, n: shape
    sigma: wanted singular values
    generator: RNG
    device, dtype: where and how to store
    Returns: matrix with exactly those singular values
    """
    r = sigma.numel()
    if r > min(m, n):
        raise ValueError("need len(sigma) <= min(m, n)")
    sigma = sigma.to(device=device, dtype=dtype)
    U = rand_orthogonal_factor(m, r, generator=generator, device=device, dtype=dtype)
    V = rand_orthogonal_factor(n, r, generator=generator, device=device, dtype=dtype)
    return (U * sigma) @ V.T


def rand_rank_deficient(
    m: int,
    n: int,
    *,
    generator: torch.Generator,
    n_zero: int = 0,
    n_small: int = 0,
    s_small: float = 1e-6,
    normalize: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    m, n: shape
    generator: RNG
    n_zero: singular values set to 0
    n_small: next ones set to s_small
    s_small: their value
    normalize: scale largest singular value to 1 first
    device, dtype: where and how to store
    Returns: Gaussian matrix with a damaged tail spectrum
    """
    if n_zero < 0 or n_small < 0:
        raise ValueError("n_zero and n_small must be nonnegative")
    r = min(m, n)
    if n_zero + n_small > r:
        raise ValueError("require n_zero + n_small <= min(m, n)")
    if s_small < 0.0:
        raise ValueError("s_small must be nonnegative")

    M = rand_gaussian(m, n, generator=generator, device=device, dtype=dtype)
    U, sigma, Vh = torch.linalg.svd(M, full_matrices=False)
    sigma = sigma.clone()
    if normalize and r and sigma[0] > 0:
        sigma = sigma / sigma[0]
    if n_small:  # sigma is descending, so the tail is the end of the array
        sigma[r - n_zero - n_small : r - n_zero] = s_small
    if n_zero:
        sigma[r - n_zero :] = 0.0
    return (U * sigma) @ Vh