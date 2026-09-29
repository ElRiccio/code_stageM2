"""Ground-truth computations (SVD, eigendecomposition, exact spectral operators)
and error norms, used to check the decomposition-free results against.

Everything runs on the input's device, in the input's dtype.
"""

from __future__ import annotations

from typing import Callable

import torch


# ----------------------------------------------------------------------------
# Reference decompositions
# ----------------------------------------------------------------------------


def reference_svd(
    M: torch.Tensor, driver: str | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    M: input matrix
    driver: cuSOLVER method (gesvd, gesvdj, gesvda)
    Returns: U, sigma (descending), V
    Note: driver only works on CUDA
    """
    U, sigma, Vh = torch.linalg.svd(M, full_matrices=False, driver=driver)
    return U, sigma, Vh.mH


def reference_eig(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    M: symmetric matrix
    Returns: eigenvalues (ascending), eigenvectors Q
    """
    return torch.linalg.eigh(M)


def numerical_rank_tol(M: torch.Tensor, sigma: torch.Tensor) -> float:
    """
    M: input matrix
    sigma: its singular values (descending)
    Returns: zero cutoff for singular values
    """
    if sigma.numel() == 0:
        return 0.0
    eps = torch.finfo(sigma.dtype).eps
    return float(max(M.shape) * eps * sigma[0])


# ----------------------------------------------------------------------------
# Reference spectral operators
# ----------------------------------------------------------------------------


def op_svd(M: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    M: input matrix
    f: scalar map, applied to singular values
    Returns: M with f applied to its spectrum
    """
    U, sigma, V = reference_svd(M)
    return (U * f(sigma)) @ V.mH


def op_eig(M: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    M: symmetric matrix
    f: scalar map, applied to eigenvalues
    Returns: M with f applied to its spectrum
    """
    lam, Q = reference_eig(M)
    return (Q * f(lam)) @ Q.mH


def spectral_coordinates(Y: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Y: input matrix
    U: left frame
    V: right frame
    Returns: coordinates of Y in the frame
    """
    return torch.einsum("ji,ji->i", U, Y @ V)


def frame_residual(Y: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Y: input matrix
    U: left frame
    V: right frame
    Returns: size of the part of Y outside the frame (0 if Y fits it)
    """
    c = spectral_coordinates(Y, U, V)
    return torch.linalg.norm(Y - (U * c) @ V.mH)


# ----------------------------------------------------------------------------
# Error norms
# ----------------------------------------------------------------------------


def frobenius_error(approx: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """
    approx: computed matrix
    exact: reference matrix
    Returns: Frobenius error
    """
    return torch.linalg.norm(approx - exact)


def relative_frobenius_error(approx: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """
    approx: computed matrix
    exact: reference matrix
    Returns: Frobenius error / norm of exact
    """
    return frobenius_error(approx, exact) / torch.linalg.norm(exact)


def spectral_error(approx: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """
    approx: computed matrix
    exact: reference matrix
    Returns: spectral-norm error
    """
    return torch.linalg.matrix_norm(approx - exact, ord=2)
