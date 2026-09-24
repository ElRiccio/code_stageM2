"""Reference computations and error norms: the decomposition-based ground
truth that decomposition-free results are compared with.

Reference decompositions and spectral operators use torch.linalg on the
device of the input; their precision is the dtype of the input matrix.
"""

from __future__ import annotations

from typing import Callable

import torch


# ----------------------------------------------------------------------------
# Reference decompositions
# ----------------------------------------------------------------------------


def reference_svd(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Thin SVD (U, sigma, V) of M with sigma descending, via torch.linalg.svd.

    Usage: U, sigma, V = reference_svd(M)
    """
    U, sigma, Vh = torch.linalg.svd(M, full_matrices=False)
    return U, sigma, Vh.mH


def reference_eig(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecomposition (lambda, Q) of a symmetric M with lambda ascending,
    via torch.linalg.eigh.

    Usage: lam, Q = reference_eig(S)
    """
    return torch.linalg.eigh(M)


def numerical_rank_tol(M: torch.Tensor, sigma: torch.Tensor) -> float:
    """Numerical-rank threshold max(m, n) * eps(dtype) * sigma_max: singular
    values at or below it read as exactly zero, the floating-point form of
    sgn(0) = 0.

    Usage: tol = numerical_rank_tol(M, sigma)
    """
    if sigma.numel() == 0:
        return 0.0
    eps = torch.finfo(sigma.dtype).eps
    return float(max(M.shape) * eps * sigma[0])


# ----------------------------------------------------------------------------
# Reference spectral operators
# ----------------------------------------------------------------------------


def op_svd(M: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """U diag(f(sigma)) V^T: the operator that applies the scalar map f to the
    singular values of M. An odd extension of the profile (see
    `cpwl.odd_extension`) maps zero singular values to zero.

    Usage: Y = op_svd(M, torch.sign)
    """
    U, sigma, V = reference_svd(M)
    return (U * f(sigma)) @ V.mH


def op_eig(M: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Q diag(f(lambda)) Q^T: the operator that applies the scalar map f to
    the eigenvalues of a symmetric M.

    Usage: Y = op_eig(S, torch.relu)
    """
    lam, Q = reference_eig(M)
    return (Q * f(lam)) @ Q.mH


def spectral_coordinates(Y: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """diag(U^T Y V): the coordinates of Y in the frame (U, V). They are the
    singular values of Y when Y = U diag(c) V^T with c >= 0.

    Usage: c = spectral_coordinates(Y, U, V)
    """
    return torch.einsum("ji,ji->i", U, Y @ V)


def frame_residual(Y: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """|| Y - U diag(diag(U^T Y V)) V^T ||_F, the part of Y outside the frame
    (U, V); it is zero exactly when Y = U diag(c) V^T.

    Usage: r = frame_residual(Y, U, V)
    """
    c = spectral_coordinates(Y, U, V)
    return torch.linalg.norm(Y - (U * c) @ V.mH)


# ----------------------------------------------------------------------------
# Error norms
# ----------------------------------------------------------------------------


def frobenius_error(approx: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """|| approx - exact ||_F.

    Usage: frobenius_error(X, N)
    """
    return torch.linalg.norm(approx - exact)


def relative_frobenius_error(approx: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """|| approx - exact ||_F / || exact ||_F.

    Usage: relative_frobenius_error(X, N)
    """
    return frobenius_error(approx, exact) / torch.linalg.norm(exact)


def spectral_error(approx: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """|| approx - exact ||_2, the spectral norm of the difference.

    Usage: spectral_error(X, N)
    """
    return torch.linalg.matrix_norm(approx - exact, ord=2)
