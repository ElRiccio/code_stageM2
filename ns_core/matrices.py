"""Random test matrices: Gaussian and symmetric draws, semi-orthogonal
factors, matrices with a prescribed spectrum, and rank-deficient or
ill-conditioned matrices.

Prescribed-spectrum and rank-deficient matrices are assembled from an
orthogonal or SVD frame with the singular values set directly. Every
function takes an explicit torch.Generator and explicit device/dtype
keywords (defaults: CPU, float32); the generator lives on `device`.
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
    """m x n matrix with i.i.d. standard normal entries (full rank almost surely).

    Usage: M = rand_gaussian(64, 32, generator=g, dtype=torch.float64)
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
    """Semi-orthogonal q x k factor (q >= k) from the QR of a Gaussian block,
    with the diagonal of R made nonnegative so equal generator states give
    equal factors.

    Usage: Q = rand_orthogonal_factor(64, 32, generator=g)
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
    """Symmetric n x n matrix 0.5 * (A + A^T) for Gaussian A; with
    `normalize` it is rescaled to spectral norm 1.

    Usage: S = rand_symmetric(64, generator=g)
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
    """U diag(sigma) V^T for random semi-orthogonal U (m x r) and V (n x r),
    r = len(sigma): a matrix whose singular values are exactly `sigma`.

    Usage: M = rand_prescribed_spectrum(64, 32, torch.linspace(1, 0.1, 32), generator=g)
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
    """Gaussian matrix whose `n_zero` smallest singular values are set to 0
    and the next `n_small` smallest to `s_small`. With `normalize`, the
    singular values are first scaled so the largest is 1.

    Usage: M = rand_rank_deficient(64, 32, generator=g, n_zero=4, n_small=2, s_small=1e-3)
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