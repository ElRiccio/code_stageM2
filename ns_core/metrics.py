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
    raise NotImplementedError


def reference_eig(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecomposition (lambda, Q) of a symmetric M, lambda ascending,
    via torch.linalg.eigh."""
    raise NotImplementedError


def numerical_rank_tol(M: torch.Tensor, sigma: torch.Tensor) -> float:
    """Numerical-rank threshold max(m, n) * eps(dtype) * sigma_max.

    The float reading of the convention sgn(0) = 0: singular values at or
    below this threshold are treated as exactly zero.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Reference spectral operators (the decomposition-based ground truth)
# ----------------------------------------------------------------------------


def op_svd(M: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Reference operator U diag(f(sigma)) V^T, singular values only.

    `f` should be an odd extension of the intended scalar profile whenever M
    may be rank-deficient (see ns_core.cpwl.odd_extension); this function
    does not extend it for you.
    """
    raise NotImplementedError


def op_eig(M: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Reference operator Q diag(f(lambda)) Q^T on a symmetric M.

    No odd extension is needed here: on Sym^n the profile is applied to the
    eigenvalues as it stands.
    """
    raise NotImplementedError


def spectral_coordinates(Y: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """diag(U^T Y V): the coordinates of Y read in the frame (U, V).

    Coincides with the singular values of Y exactly when the map that
    produced Y keeps the frame of its argument and the underlying scalar
    profile is nondecreasing on [0, 1].
    """
    raise NotImplementedError


def frame_residual(Y: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """|| Y - U diag(diag(U^T Y V)) V^T ||_F.

    Zero iff Y lives entirely in the frame (U, V); a nonzero residual flags
    that an approximate map (e.g. a truncated NS iteration) has leaked
    outside the expected spectral coordinates.
    """
    raise NotImplementedError


# ----------------------------------------------------------------------------
# Error norms
# ----------------------------------------------------------------------------


def frobenius_error(approx: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """|| approx - exact ||_F."""
    raise NotImplementedError


def relative_frobenius_error(approx: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """|| approx - exact ||_F / || exact ||_F.

    The relative-error quantity the reviewer asks be reported in the
    quantitative comparison table, alongside iteration/evaluation counts and
    timing (see experiments.svd_comparison).
    """
    raise NotImplementedError


def spectral_error(approx: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """|| approx - exact ||_2 (exact spectral norm of the difference)."""
    raise NotImplementedError
