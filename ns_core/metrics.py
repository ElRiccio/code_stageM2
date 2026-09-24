"""Reference computations and error norms: the ground truth every experiment
compares its decomposition-free result against.

Reference decompositions and reference spectral operators go through
torch.linalg exclusively. Precision (float32 vs float64) for a reference
computation is a per-experiment choice, passed in via the dtype of the
input matrix; nothing here hardcodes it.
"""

from __future__ import annotations

from typing import Callable

import torch


# ----------------------------------------------------------------------------
# Reference decompositions
# ----------------------------------------------------------------------------


def reference_svd(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Thin SVD (U, sigma, V) of M, sigma descending, via torch.linalg.svd."""
    U, sigma, Vh = torch.linalg.svd(M, full_matrices=False)
    return U, sigma, Vh.mH


def reference_eig(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecomposition (lambda, Q) of a symmetric M, lambda ascending,
    via torch.linalg.eigh."""
    return torch.linalg.eigh(M)


def numerical_rank_tol(M: torch.Tensor, sigma: torch.Tensor) -> float:
    """Numerical-rank threshold max(m, n) * eps(dtype) * sigma_max.

    The float reading of the convention sgn(0) = 0: singular values at or
    below this threshold are treated as exactly zero.
    """
    if sigma.numel() == 0:
        return 0.0
    eps = torch.finfo(sigma.dtype).eps
    return float(max(M.shape) * eps * sigma[0])


# ----------------------------------------------------------------------------
# Reference spectral operators (the decomposition-based ground truth)
# ----------------------------------------------------------------------------


def op_svd(M: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Reference operator U diag(f(sigma)) V^T, singular values only.

    `f` should be an odd extension of the intended scalar profile whenever M
    may be rank-deficient (see ns_core.cpwl.odd_extension); this function
    does not extend it for you.
    """
    U, sigma, V = reference_svd(M)
    return (U * f(sigma)) @ V.mH


def op_eig(M: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Reference operator Q diag(f(lambda)) Q^T on a symmetric M.

    No odd extension is needed here: on Sym^n the profile is applied to the
    eigenvalues as it stands.
    """
    lam, Q = reference_eig(M)
    return (Q * f(lam)) @ Q.mH


def spectral_coordinates(Y: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """diag(U^T Y V): the coordinates of Y read in the frame (U, V).

    Coincides with the singular values of Y exactly when the map that
    produced Y keeps the frame of its argument and the underlying scalar
    profile is nondecreasing on [0, 1].
    """
    return torch.einsum("ji,ji->i", U, Y @ V)


def frame_residual(Y: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """|| Y - U diag(diag(U^T Y V)) V^T ||_F.

    Zero iff Y lives entirely in the frame (U, V); a nonzero residual flags
    that an approximate map (e.g. a truncated NS iteration) has leaked
    outside the expected spectral coordinates.
    """
    c = spectral_coordinates(Y, U, V)
    return torch.linalg.norm(Y - (U * c) @ V.mH)


# ----------------------------------------------------------------------------
# Error norms
# ----------------------------------------------------------------------------


def frobenius_error(approx: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """|| approx - exact ||_F."""
    return torch.linalg.norm(approx - exact)


def relative_frobenius_error(approx: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """|| approx - exact ||_F / || exact ||_F.

    The relative-error quantity the reviewer asks be reported in the
    quantitative comparison table, alongside iteration/evaluation counts and
    timing.
    """
    return frobenius_error(approx, exact) / torch.linalg.norm(exact)


def spectral_error(approx: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """|| approx - exact ||_2 (exact spectral norm of the difference)."""
    return torch.linalg.matrix_norm(approx - exact, ord=2)
