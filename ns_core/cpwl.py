"""Piecewise-linear (CPWL) profiles, and how to apply them to a matrix's spectrum
using only a matrix sign function `sgn` and matrix products (no SVD).

Two families: rectangular versions (act on singular values, check against
`metrics.op_svd`) and symmetric versions (act on eigenvalues, check against
`metrics.op_eig`). `sgn` can be exact or a Newton-Schulz approximation.

`odd_extension` turns a profile on [0, infinity) into an odd one, which is what
acts on singular values.
"""

from __future__ import annotations

from typing import Callable

import torch

from ns_core.sign_map import msgn, sgn_exact


# ----------------------------------------------------------------------------
# Scalar profiles
# ----------------------------------------------------------------------------


def relu(x: torch.Tensor) -> torch.Tensor:
    """
    x: points
    Returns: max(x, 0)
    """
    return torch.clamp(x, min=0.0)


def leaky_relu(x: torch.Tensor, a: float) -> torch.Tensor:
    """
    x: points
    a: negative-side slope (< 1)
    Returns: leaky ReLU of x
    """
    return torch.maximum(x, a * x)


def capped_leaky_relu(x: torch.Tensor, a: float, beta: float) -> torch.Tensor:
    """
    x: points
    a: negative-side slope (< 1)
    beta: cap (> 0)
    Returns: leaky ReLU of x, capped at beta
    """
    return torch.clamp(leaky_relu(x, a), max=beta)


def clip(x: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """
    x: points
    alpha: lower bound
    beta: upper bound (> alpha)
    Returns: x clipped to [alpha, beta]
    """
    if not alpha < beta:
        raise ValueError("require alpha < beta")
    return torch.clamp(x, alpha, beta)


def soft_threshold(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    x: points
    gamma: threshold (> 0)
    Returns: x shrunk toward 0 by gamma
    """
    if gamma <= 0.0:
        raise ValueError("require gamma > 0")
    return torch.sign(x) * torch.clamp(torch.abs(x) - gamma, min=0.0)


def leaky_clip(x: torch.Tensor, a: float, mu: float) -> torch.Tensor:
    """
    x: points
    a: outer slope
    mu: half-width of the steep middle
    Returns: odd leaky clip of x
    """
    return a * x + 0.5 * (1.0 - a) * (torch.abs(x + mu) - torch.abs(x - mu))


def odd_extension(f: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    f: profile on [0, infinity)
    Returns: odd version of f, sign(x) f(|x|)
    Note: always 0 at 0, whatever f(0) is
    """

    def g(x: torch.Tensor) -> torch.Tensor:
        return sgn_exact(x) * f(torch.abs(x))

    return g


class PiecewiseLinearProfile:
    """A piecewise-linear function through given points, in ReLU and sign form.

    knots: increasing breakpoints
    vals: function value at each knot
    Attributes: c (slope per cell), w (slope jump per interior knot),
    interior (interior knots), eta, theta, kappa (constants of the two forms)
    Note: stored in float64
    """

    def __init__(self, knots: torch.Tensor, vals: torch.Tensor):
        knots = torch.as_tensor(knots, dtype=torch.float64).reshape(-1)
        vals = torch.as_tensor(vals, dtype=torch.float64).reshape(-1)
        if knots.numel() < 2:
            raise ValueError("need at least two knots")
        if knots.shape != vals.shape:
            raise ValueError("knots and vals must have the same length")
        if torch.any(torch.diff(knots) <= 0.0):
            raise ValueError("knots must be strictly increasing")

        h = torch.diff(knots)
        self.knots = knots
        self.vals = vals
        self.c = torch.diff(vals) / h
        self.w = torch.diff(self.c)
        self.interior = knots[1:-1]
        self.eta = float(vals[0] - self.c[0] * knots[0])
        self.theta = float(0.5 * (self.c[0] + self.c[-1]))
        self.kappa = float(self.eta - 0.5 * torch.sum(self.w * self.interior))

    def eval_relu_form(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: points
        Returns: profile value, via ReLUs
        """
        out = self.eta + float(self.c[0]) * x
        for wi, xi in zip(self.w.tolist(), self.interior.tolist()):
            out = out + wi * relu(x - xi)
        return out

    def eval_sign_form(self, x: torch.Tensor, sgn) -> torch.Tensor:
        """
        x: points
        sgn: scalar sign function
        Returns: profile value, via signs (the form spline_map_sign lifts)
        """
        out = self.kappa + self.theta * x
        for wi, xi in zip(self.w.tolist(), self.interior.tolist()):
            t = x - xi
            out = out + 0.5 * wi * t * sgn(t)
        return out


# ----------------------------------------------------------------------------
# Rectangular sign forms, R^{m x n}
# ----------------------------------------------------------------------------


def clip_map(M: torch.Tensor, alpha: float, beta: float, sgn: msgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """
    M: input matrix
    alpha: lower bound
    beta: upper bound (> alpha)
    sgn: matrix sign function
    N: precomputed sgn(M)
    Returns: M with its singular values clipped to [alpha, beta]
    Note: calls sgn on every shifted matrix, even for non-positive bounds
    """
    if not alpha < beta:
        raise ValueError("require alpha < beta")
    if N is None:
        N = sgn(M)
    I = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    Ma = alpha * N - M
    Mb = beta * N - M
    core = (alpha + beta) * I + Ma @ sgn(Ma).mH - Mb @ sgn(Mb).mH
    return 0.5 * (core @ N)


def hinge_map(M: torch.Tensor, mu: float, sgn: msgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """
    M: input matrix
    mu: hinge position
    sgn: matrix sign function
    N: precomputed sgn(M)
    Returns: matrix hinge of M at mu
    """
    if N is None:
        N = sgn(M)
    Mmu = mu * N - M
    return 0.5 * (Mmu @ sgn(Mmu).mH @ N - Mmu)


def hinge_map_neg(M: torch.Tensor, mu: float, sgn: msgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """
    M: input matrix
    mu: hinge position (<= 0)
    sgn: matrix sign function
    N: precomputed sgn(M)
    Returns: matrix hinge of M at mu, closed form
    """
    if mu > 0.0:
        raise ValueError("the closed form is valid for mu <= 0 only")
    if N is None:
        N = sgn(M)
    return M - mu * N


def soft_map(M: torch.Tensor, gamma: float, sgn: msgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """
    M: input matrix
    gamma: threshold (> 0)
    sgn: matrix sign function
    N: precomputed sgn(M)
    Returns: M with singular values soft-thresholded (same as hinge_map at gamma)
    """
    if gamma <= 0.0:
        raise ValueError("require gamma > 0")
    return hinge_map(M, gamma, sgn, N)


def spline_map(M: torch.Tensor, spline: PiecewiseLinearProfile, sgn: msgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """
    M: input matrix
    spline: profile to apply
    sgn: matrix sign function
    N: precomputed sgn(M)
    Returns: profile applied to M's singular values, via the ReLU form
    """
    if N is None:
        N = sgn(M)
    out = spline.eta * N + float(spline.c[0]) * M
    for wi, xi in zip(spline.w.tolist(), spline.interior.tolist()):
        out = out + wi * hinge_map(M, xi, sgn, N)
    return out


def spline_map_sign(M: torch.Tensor, spline: PiecewiseLinearProfile, sgn: msgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """
    M: input matrix
    spline: profile to apply
    sgn: matrix sign function
    N: precomputed sgn(M)
    Returns: profile applied to M's singular values, via the sign form
    """
    if N is None:
        N = sgn(M)
    out = spline.kappa * N + spline.theta * M
    for wi, xi in zip(spline.w.tolist(), spline.interior.tolist()):
        Mx = xi * N - M
        out = out + 0.5 * wi * (Mx @ sgn(Mx).mH @ N)
    return out


def matrix_modulus(M: torch.Tensor, mu: float, sgn: msgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """
    M: input matrix
    mu: position
    sgn: matrix sign function
    N: precomputed sgn(M)
    Returns: matrix modulus of M at mu (building block of the leaky forms)
    """
    if N is None:
        N = sgn(M)
    Mmu = mu * N - M
    return Mmu @ sgn(Mmu).mH @ N


def leaky_relu_map(M: torch.Tensor, a: float, sgn: msgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """
    M: input matrix
    a, sgn, N: unused, kept so all maps share one signature
    Returns: M itself (leaky ReLU is the identity on singular values)
    """
    return M


def capped_leaky_relu_map(M: torch.Tensor, a: float, beta: float, sgn: msgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """
    M: input matrix
    a: negative-side slope (unused here)
    beta: cap
    sgn: matrix sign function
    N: precomputed sgn(M)
    Returns: M with singular values leaky-ReLU'd and capped at beta
    """
    if N is None:
        N = sgn(M)
    return 0.5 * (beta * N + M - matrix_modulus(M, beta, sgn, N))


def leaky_clip_map(M: torch.Tensor, a: float, mu: float, sgn: msgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """
    M: input matrix
    a: outer slope
    mu: half-width of the steep middle
    sgn: matrix sign function
    N: precomputed sgn(M)
    Returns: M with singular values passed through the leaky clip
    """
    if N is None:
        N = sgn(M)
    return a * M + 0.5 * (1.0 - a) * (M + mu * N - matrix_modulus(M, mu, sgn, N))


# ----------------------------------------------------------------------------
# Symmetric sign forms, Sym^n
# ----------------------------------------------------------------------------


def matrix_abs(A: torch.Tensor, sgn: msgn) -> torch.Tensor:
    """
    A: symmetric matrix
    sgn: matrix sign function
    Returns: matrix absolute value |A|
    """
    return A @ sgn(A).mH


def clip_map_sym(M: torch.Tensor, alpha: float, beta: float, sgn: msgn) -> torch.Tensor:
    """
    M: symmetric matrix
    alpha: lower bound
    beta: upper bound
    sgn: matrix sign function
    Returns: M with eigenvalues clipped to [alpha, beta]
    """
    I = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    return 0.5 * ((alpha + beta) * I + matrix_abs(M - alpha * I, sgn) - matrix_abs(M - beta * I, sgn))


def soft_map_sym(M: torch.Tensor, gamma: float, sgn: msgn) -> torch.Tensor:
    """
    M: symmetric matrix
    gamma: threshold
    sgn: matrix sign function
    Returns: M with eigenvalues soft-thresholded
    """
    I = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    return M + 0.5 * (matrix_abs(M - gamma * I, sgn) - matrix_abs(M + gamma * I, sgn))


def leaky_relu_map_sym(M: torch.Tensor, a: float, sgn: msgn) -> torch.Tensor:
    """
    M: symmetric matrix
    a: negative-side slope
    sgn: matrix sign function
    Returns: M with eigenvalues leaky-ReLU'd
    """
    return 0.5 * ((1.0 + a) * M + (1.0 - a) * matrix_abs(M, sgn))


def capped_leaky_relu_map_sym(M: torch.Tensor, a: float, beta: float, sgn: msgn) -> torch.Tensor:
    """
    M: symmetric matrix
    a: negative-side slope
    beta: cap
    sgn: matrix sign function
    Returns: M with eigenvalues leaky-ReLU'd and capped at beta
    """
    I = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    return 0.5 * (beta * I + a * M + (1.0 - a) * matrix_abs(M, sgn) - matrix_abs(M - beta * I, sgn))


def leaky_clip_map_sym(M: torch.Tensor, a: float, mu: float, sgn: msgn) -> torch.Tensor:
    """
    M: symmetric matrix
    a: outer slope
    mu: half-width of the steep middle
    sgn: matrix sign function
    Returns: M with eigenvalues passed through the leaky clip
    """
    I = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    return a * M + 0.5 * (1.0 - a) * (matrix_abs(M + mu * I, sgn) - matrix_abs(M - mu * I, sgn))
