"""The catalogue of continuous piecewise-linear (CPWL) scalar profiles from
Chapter 5, and their decomposition-free sign forms: the matrix expressions
that evaluate each profile spectrally using only a Sgn callable (exact or a
truncated NS surrogate, see ns_core.sign_map) and matrix products, never an
SVD or eigendecomposition of their own.

Two families of sign forms are given for every profile that admits both:
  - the rectangular forms, acting on M in R^{m x n} via the singular-value
    frame (used with ns_core.metrics.op_svd as reference);
  - the symmetric forms, acting on M in Sym^n via the eigenvalue frame (used
    with ns_core.metrics.op_eig as reference), built from the matrix
    absolute value |A| = A Sgn(A)^T.

A scalar profile that is not odd (e.g. plain ReLU, clip with alpha != -beta)
must be extended to an odd function before it can be read off a singular
value frame at all, since singular values are nonnegative and Sgn forms
implicitly assume an odd extension; `odd_extension` does that.
"""

from __future__ import annotations

from typing import Callable

import torch

from ns_core.sign_map import Sgn, sgn_exact


# ----------------------------------------------------------------------------
# Scalar profiles
# ----------------------------------------------------------------------------


def relu(x: torch.Tensor) -> torch.Tensor:
    """max(x, 0)."""
    return torch.clamp(x, min=0.0)


def leaky_relu(x: torch.Tensor, a: float) -> torch.Tensor:
    """max(x, a x), for a < 1."""
    return torch.maximum(x, a * x)


def capped_leaky_relu(x: torch.Tensor, a: float, beta: float) -> torch.Tensor:
    """min(leaky_relu(x, a), beta), for a < 1 and beta > 0."""
    return torch.clamp(leaky_relu(x, a), max=beta)


def clip(x: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Exact clipping onto [alpha, beta]. Requires alpha < beta."""
    if not alpha < beta:
        raise ValueError("require alpha < beta")
    return torch.clamp(x, alpha, beta)


def soft_threshold(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """sign(x) * max(|x| - gamma, 0). Requires gamma > 0."""
    if gamma <= 0.0:
        raise ValueError("require gamma > 0")
    return torch.sign(x) * torch.clamp(torch.abs(x) - gamma, min=0.0)


def leaky_clip(x: torch.Tensor, a: float, mu: float) -> torch.Tensor:
    """a x + (1-a)/2 (|x + mu| - |x - mu|). Odd by construction."""
    return a * x + 0.5 * (1.0 - a) * (torch.abs(x + mu) - torch.abs(x - mu))


def odd_extension(f: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    """x -> sign(x) f(|x|): the odd extension of a scalar profile defined on
    [0, infinity).

    Vanishes at the origin regardless of f(0), which is exactly the property
    that makes the induced spectral operator well posed on a rank-deficient
    argument (a zero singular value maps to zero, not to f(0)).
    """

    def g(x: torch.Tensor) -> torch.Tensor:
        return sgn_exact(x) * f(torch.abs(x))

    return g


class PiecewiseLinearProfile:
    """A linear spline on a partition, with its ReLU and sign representations.

    Attributes set at construction: `knots`, `vals`, `c` (per-cell slopes),
    `w` (slope jumps at interior knots), `interior` (interior knots), and the
    three scalars `eta`, `theta`, `kappa` the sign representation is written
    in terms of.
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
        """eta + c[0] x + sum_i w_i ReLU(x - interior_i): the reference value."""
        out = self.eta + float(self.c[0]) * x
        for wi, xi in zip(self.w.tolist(), self.interior.tolist()):
            out = out + wi * relu(x - xi)
        return out

    def eval_sign_form(self, x: torch.Tensor, sgn) -> torch.Tensor:
        """kappa + theta x + sum_i (w_i / 2) (x - interior_i) sgn(x - interior_i):
        the form that lifts to matrices via `spline_map`/`spline_map_sign`."""
        out = self.kappa + self.theta * x
        for wi, xi in zip(self.w.tolist(), self.interior.tolist()):
            t = x - xi
            out = out + 0.5 * wi * t * sgn(t)
        return out


# ----------------------------------------------------------------------------
# Rectangular sign forms, R^{m x n}
# ----------------------------------------------------------------------------


def clip_map(M: torch.Tensor, alpha: float, beta: float, sgn: Sgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """(1/2) [(alpha+beta) I + M_a Sgn(M_a)^T - M_b Sgn(M_b)^T] Sgn(M),
    with M_a = alpha Sgn(M) - M and M_b = beta Sgn(M) - M.

    `N`, if given, is a precomputed Sgn(M); passing it avoids recomputing the
    outer sign map when this is combined with other forms that also need it.
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


def hinge_map(M: torch.Tensor, mu: float, sgn: Sgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """(1/2) [M_mu Sgn(M_mu)^T Sgn(M) - M_mu], M_mu = mu Sgn(M) - M.

    The matrix hinge at threshold mu; soft-thresholding at gamma > 0 is the
    special case hinge_map(M, gamma, ...) (see soft_map)."""
    if N is None:
        N = sgn(M)
    Mmu = mu * N - M
    return 0.5 * (Mmu @ sgn(Mmu).mH @ N - Mmu)


def hinge_map_neg(M: torch.Tensor, mu: float, sgn: Sgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """Closed form M - mu Sgn(M), valid only for mu <= 0."""
    if mu > 0.0:
        raise ValueError("the closed form is valid for mu <= 0 only")
    if N is None:
        N = sgn(M)
    return M - mu * N


def soft_map(M: torch.Tensor, gamma: float, sgn: Sgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """The matrix hinge at a positive threshold: soft_map = hinge_map(M, gamma, ...)
    for gamma > 0."""
    if gamma <= 0.0:
        raise ValueError("require gamma > 0")
    return hinge_map(M, gamma, sgn, N)


def spline_map(M: torch.Tensor, spline: PiecewiseLinearProfile, sgn: Sgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """eta Sgn(M) + c[0] M + sum_i w_i hinge_map(M, interior_i, sgn, N):
    the ReLU-form lift of an arbitrary PiecewiseLinearProfile."""
    if N is None:
        N = sgn(M)
    out = spline.eta * N + float(spline.c[0]) * M
    for wi, xi in zip(spline.w.tolist(), spline.interior.tolist()):
        out = out + wi * hinge_map(M, xi, sgn, N)
    return out


def spline_map_sign(M: torch.Tensor, spline: PiecewiseLinearProfile, sgn: Sgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """kappa Sgn(M) + theta M + sum_i (w_i/2) M_i Sgn(M_i)^T Sgn(M), M_i =
    interior_i Sgn(M) - M: the sign-form lift, an independent route to the
    same result as spline_map (their agreement is itself worth checking)."""
    if N is None:
        N = sgn(M)
    out = spline.kappa * N + spline.theta * M
    for wi, xi in zip(spline.w.tolist(), spline.interior.tolist()):
        Mx = xi * N - M
        out = out + 0.5 * wi * (Mx @ sgn(Mx).mH @ N)
    return out


def matrix_modulus(M: torch.Tensor, mu: float, sgn: Sgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """Mod_mu(M) = S_mu Sgn(S_mu)^T Sgn(M), S_mu = mu Sgn(M) - M.

    The rectangular building block behind the leaky and capped-leaky forms
    below."""
    if N is None:
        N = sgn(M)
    Mmu = mu * N - M
    return Mmu @ sgn(Mmu).mH @ N


def leaky_relu_map(M: torch.Tensor, a: float, sgn: Sgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """On R^{m x n}, the sign form does not see the leak: leaky_relu_map(M, a, ...) = M."""
    return M


def capped_leaky_relu_map(M: torch.Tensor, a: float, beta: float, sgn: Sgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """(1/2) [beta Sgn(M) + M - matrix_modulus(M, beta, sgn, N)]."""
    if N is None:
        N = sgn(M)
    return 0.5 * (beta * N + M - matrix_modulus(M, beta, sgn, N))


def leaky_clip_map(M: torch.Tensor, a: float, mu: float, sgn: Sgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """a M + (1-a)/2 [M + mu Sgn(M) - matrix_modulus(M, mu, sgn, N)]."""
    if N is None:
        N = sgn(M)
    return a * M + 0.5 * (1.0 - a) * (M + mu * N - matrix_modulus(M, mu, sgn, N))


# ----------------------------------------------------------------------------
# Symmetric sign forms, Sym^n
# ----------------------------------------------------------------------------


def matrix_abs(A: torch.Tensor, sgn: Sgn) -> torch.Tensor:
    """|A| = A Sgn(A)^T, for symmetric A: the symmetric-layer building block
    every form below is written in terms of."""
    return A @ sgn(A).mH


def clip_map_sym(M: torch.Tensor, alpha: float, beta: float, sgn: Sgn) -> torch.Tensor:
    """(1/2) [(alpha+beta) I + |M - alpha I| - |M - beta I|]."""
    I = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    return 0.5 * ((alpha + beta) * I + matrix_abs(M - alpha * I, sgn) - matrix_abs(M - beta * I, sgn))


def soft_map_sym(M: torch.Tensor, gamma: float, sgn: Sgn) -> torch.Tensor:
    """M + (1/2) [|M - gamma I| - |M + gamma I|]."""
    I = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    return M + 0.5 * (matrix_abs(M - gamma * I, sgn) - matrix_abs(M + gamma * I, sgn))


def leaky_relu_map_sym(M: torch.Tensor, a: float, sgn: Sgn) -> torch.Tensor:
    """(1/2) [(1+a) M + (1-a) |M|]."""
    return 0.5 * ((1.0 + a) * M + (1.0 - a) * matrix_abs(M, sgn))


def capped_leaky_relu_map_sym(M: torch.Tensor, a: float, beta: float, sgn: Sgn) -> torch.Tensor:
    """(1/2) [beta I + a M + (1-a) |M| - |M - beta I|]."""
    I = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    return 0.5 * (beta * I + a * M + (1.0 - a) * matrix_abs(M, sgn) - matrix_abs(M - beta * I, sgn))


def leaky_clip_map_sym(M: torch.Tensor, a: float, mu: float, sgn: Sgn) -> torch.Tensor:
    """a M + (1-a)/2 [|M + mu I| - |M - mu I|]."""
    I = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    return a * M + 0.5 * (1.0 - a) * (matrix_abs(M + mu * I, sgn) - matrix_abs(M - mu * I, sgn))
