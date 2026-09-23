"""Random test matrices: plain generators and rank/conditioning-controlled ones.

Every generator here is a reference construction: rank-deficient and
ill-conditioned matrices are built by taking an exact SVD and overwriting
singular values directly. That is legitimate for building a test instance,
but it is a different computation from the decomposition-free algorithms
these matrices are used to test (see ns_core.sign_map, ns_core.cpwl), and
the two should never be confused with one another in an experiment.

Every function takes an explicit torch.Generator for reproducibility rather
than relying on global RNG state, and explicit device/dtype kwargs with
CPU/float32 defaults.
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
    """Plain m x n Gaussian matrix. Almost surely full rank, spectrum unprescribed."""
    raise NotImplementedError


def rand_orthogonal_factor(
    q: int,
    k: int,
    *,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Semi-orthogonal q x k factor (q >= k) from the QR of a Gaussian block.

    The QR sign ambiguity is pinned (diagonal of R made nonnegative) so that
    runs with the same generator state reproduce exactly.
    """
    raise NotImplementedError


def rand_symmetric(
    n: int,
    *,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    normalize: bool = True,
) -> torch.Tensor:
    """Symmetric n x n test matrix, 0.5 * (A + A^T) for Gaussian A.

    If `normalize`, rescaled so that its exact spectral norm is 1, which is
    the convention the rest of the library assumes (every knot/threshold
    passed to a profile is then read directly against the spectrum).
    """
    raise NotImplementedError


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
    """Gaussian matrix with its smallest singular values overwritten.

    Built as: draw a Gaussian matrix, take its exact SVD, set the `n_small`
    smallest nonzero singular values to `s_small` and the `n_zero` smallest
    to exactly 0, then reassemble. This is the rank-deficiency /
    conditioning knob the reviewer asks be swept systematically (several
    ranks, several values of the smallest nonzero singular value) rather
    than fixed at "one singular value set to zero".

    Requires n_zero + n_small <= min(m, n). If `normalize`, the singular
    values are scaled first so the largest is 1, before the overwrite.
    """
    raise NotImplementedError
