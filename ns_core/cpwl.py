"""Continuous piecewise-linear (CPWL) scalar profiles and their
decomposition-free sign forms: matrix expressions that evaluate each profile
spectrally from a matrix sign callable `sgn` (of type `sign_map.msgn`, exact
or a truncated NS surrogate) and matrix products.

Two families of sign forms are provided:
  - rectangular forms on M in R^{m x n}, acting on singular values (reference:
    `metrics.op_svd`);
  - symmetric forms on M in Sym^n, acting on eigenvalues (reference:
    `metrics.op_eig`), written with the matrix absolute value
    |A| = A msgn(A)^T.

`odd_extension` turns a profile defined on [0, infinity) into the odd map
x -> sign(x) f(|x|), the form that reads singular values.
"""

from __future__ import annotations

from typing import Callable

import torch

from ns_core.sign_map import msgn, sgn_exact


# ----------------------------------------------------------------------------
# Scalar profiles
# ----------------------------------------------------------------------------


def relu(x: torch.Tensor) -> torch.Tensor:
    """max(x, 0).

    Usage: relu(torch.linspace(-1, 1, 5))
    """
    return torch.clamp(x, min=0.0)


def leaky_relu(x: torch.Tensor, a: float) -> torch.Tensor:
    """max(x, a x) for a slope a < 1.

    Usage: leaky_relu(x, a=0.1)
    """
    return torch.maximum(x, a * x)


def capped_leaky_relu(x: torch.Tensor, a: float, beta: float) -> torch.Tensor:
    """min(leaky_relu(x, a), beta) for a < 1 and a cap beta > 0.

    Usage: capped_leaky_relu(x, a=0.1, beta=0.5)
    """
    return torch.clamp(leaky_relu(x, a), max=beta)


def clip(x: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """x clipped onto [alpha, beta], alpha < beta.

    Usage: clip(x, alpha=-0.5, beta=0.5)
    """
    if not alpha < beta:
        raise ValueError("require alpha < beta")
    return torch.clamp(x, alpha, beta)


def soft_threshold(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """sign(x) max(|x| - gamma, 0) for a threshold gamma > 0.

    Usage: soft_threshold(x, gamma=0.2)
    """
    if gamma <= 0.0:
        raise ValueError("require gamma > 0")
    return torch.sign(x) * torch.clamp(torch.abs(x) - gamma, min=0.0)


def leaky_clip(x: torch.Tensor, a: float, mu: float) -> torch.Tensor:
    """a x + (1 - a)/2 (|x + mu| - |x - mu|), an odd map.

    Usage: leaky_clip(x, a=0.1, mu=0.5)
    """
    return a * x + 0.5 * (1.0 - a) * (torch.abs(x + mu) - torch.abs(x - mu))


def odd_extension(f: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    """The odd extension x -> sign(x) f(|x|) of a profile f on [0, infinity).
    It vanishes at the origin for any f(0), so zero singular values map to
    zero.

    Usage: g = odd_extension(relu)
    """

    def g(x: torch.Tensor) -> torch.Tensor:
        return sgn_exact(x) * f(torch.abs(x))

    return g


class PiecewiseLinearProfile:
    """A linear spline on a partition, with its ReLU and sign representations.

    Built from strictly increasing `knots` and values `vals` at the knots,
    stored in float64. Attributes: `knots`, `vals`, `c` (per-cell slopes),
    `w` (slope jumps at the interior knots), `interior` (interior knots) and
    the scalars `eta`, `theta`, `kappa` of the sign representation.

    Usage: spline = PiecewiseLinearProfile(torch.tensor([0., 1., 2.]), torch.tensor([0., 1., 1.5]))
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
        """eta + c[0] x + sum_i w_i ReLU(x - interior_i).

        Usage: spline.eval_relu_form(x)
        """
        out = self.eta + float(self.c[0]) * x
        for wi, xi in zip(self.w.tolist(), self.interior.tolist()):
            out = out + wi * relu(x - xi)
        return out

    def eval_sign_form(self, x: torch.Tensor, sgn) -> torch.Tensor:
        """kappa + theta x + sum_i (w_i / 2) (x - interior_i) sgn(x - interior_i),
        the form that lifts to matrices in `spline_map_sign`.

        Usage: spline.eval_sign_form(x, sgn_exact)
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
    """(1/2) [(alpha + beta) I + M_a msgn(M_a)^T - M_b msgn(M_b)^T] msgn(M),
    with M_a = alpha msgn(M) - M and M_b = beta msgn(M) - M. `N` is an
    optional precomputed msgn(M).

    Usage: Y = clip_map(M, -0.5, 0.5, make_sgn_ns(D=2, n_iters=8))
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
    """(1/2) [M_mu msgn(M_mu)^T msgn(M) - M_mu] with M_mu = mu msgn(M) - M:
    the matrix hinge at threshold mu. `N` is an optional precomputed msgn(M).

    Usage: Y = hinge_map(M, 0.3, sgn)
    """
    if N is None:
        N = sgn(M)
    Mmu = mu * N - M
    return 0.5 * (Mmu @ sgn(Mmu).mH @ N - Mmu)


def hinge_map_neg(M: torch.Tensor, mu: float, sgn: msgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """M - mu msgn(M): the closed form of the hinge at a threshold mu <= 0.

    Usage: Y = hinge_map_neg(M, -0.3, sgn)
    """
    if mu > 0.0:
        raise ValueError("the closed form is valid for mu <= 0 only")
    if N is None:
        N = sgn(M)
    return M - mu * N


def soft_map(M: torch.Tensor, gamma: float, sgn: msgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """Matrix soft-thresholding at gamma > 0, i.e. hinge_map(M, gamma, sgn, N).

    Usage: Y = soft_map(M, 0.2, sgn)
    """
    if gamma <= 0.0:
        raise ValueError("require gamma > 0")
    return hinge_map(M, gamma, sgn, N)


def spline_map(M: torch.Tensor, spline: PiecewiseLinearProfile, sgn: msgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """eta msgn(M) + c[0] M + sum_i w_i hinge_map(M, interior_i, sgn, N): the
    ReLU-form lift of a PiecewiseLinearProfile.

    Usage: Y = spline_map(M, spline, sgn)
    """
    if N is None:
        N = sgn(M)
    out = spline.eta * N + float(spline.c[0]) * M
    for wi, xi in zip(spline.w.tolist(), spline.interior.tolist()):
        out = out + wi * hinge_map(M, xi, sgn, N)
    return out


def spline_map_sign(M: torch.Tensor, spline: PiecewiseLinearProfile, sgn: msgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """kappa msgn(M) + theta M + sum_i (w_i / 2) M_i msgn(M_i)^T msgn(M) with
    M_i = interior_i msgn(M) - M: the sign-form lift of a
    PiecewiseLinearProfile.

    Usage: Y = spline_map_sign(M, spline, sgn)
    """
    if N is None:
        N = sgn(M)
    out = spline.kappa * N + spline.theta * M
    for wi, xi in zip(spline.w.tolist(), spline.interior.tolist()):
        Mx = xi * N - M
        out = out + 0.5 * wi * (Mx @ sgn(Mx).mH @ N)
    return out


def matrix_modulus(M: torch.Tensor, mu: float, sgn: msgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """Mod_mu(M) = S_mu msgn(S_mu)^T msgn(M) with S_mu = mu msgn(M) - M, the
    building block of the leaky and capped-leaky forms.

    Usage: Y = matrix_modulus(M, 0.5, sgn)
    """
    if N is None:
        N = sgn(M)
    Mmu = mu * N - M
    return Mmu @ sgn(Mmu).mH @ N


def leaky_relu_map(M: torch.Tensor, a: float, sgn: msgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """The leaky ReLU on R^{m x n}: the sign form reduces to the identity, so
    this returns M.

    Usage: Y = leaky_relu_map(M, 0.1, sgn)
    """
    return M


def capped_leaky_relu_map(M: torch.Tensor, a: float, beta: float, sgn: msgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """(1/2) [beta msgn(M) + M - matrix_modulus(M, beta, sgn, N)].

    Usage: Y = capped_leaky_relu_map(M, 0.1, 0.5, sgn)
    """
    if N is None:
        N = sgn(M)
    return 0.5 * (beta * N + M - matrix_modulus(M, beta, sgn, N))


def leaky_clip_map(M: torch.Tensor, a: float, mu: float, sgn: msgn, N: torch.Tensor | None = None) -> torch.Tensor:
    """a M + (1 - a)/2 [M + mu msgn(M) - matrix_modulus(M, mu, sgn, N)].

    Usage: Y = leaky_clip_map(M, 0.1, 0.5, sgn)
    """
    if N is None:
        N = sgn(M)
    return a * M + 0.5 * (1.0 - a) * (M + mu * N - matrix_modulus(M, mu, sgn, N))


# ----------------------------------------------------------------------------
# Symmetric sign forms, Sym^n
# ----------------------------------------------------------------------------


def matrix_abs(A: torch.Tensor, sgn: msgn) -> torch.Tensor:
    """|A| = A msgn(A)^T for symmetric A, the building block of the symmetric
    forms.

    Usage: B = matrix_abs(S, sgn)
    """
    return A @ sgn(A).mH


def clip_map_sym(M: torch.Tensor, alpha: float, beta: float, sgn: msgn) -> torch.Tensor:
    """(1/2) [(alpha + beta) I + |M - alpha I| - |M - beta I|].

    Usage: Y = clip_map_sym(S, -0.5, 0.5, sgn)
    """
    I = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    return 0.5 * ((alpha + beta) * I + matrix_abs(M - alpha * I, sgn) - matrix_abs(M - beta * I, sgn))


def soft_map_sym(M: torch.Tensor, gamma: float, sgn: msgn) -> torch.Tensor:
    """M + (1/2) [|M - gamma I| - |M + gamma I|].

    Usage: Y = soft_map_sym(S, 0.2, sgn)
    """
    I = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    return M + 0.5 * (matrix_abs(M - gamma * I, sgn) - matrix_abs(M + gamma * I, sgn))


def leaky_relu_map_sym(M: torch.Tensor, a: float, sgn: msgn) -> torch.Tensor:
    """(1/2) [(1 + a) M + (1 - a) |M|].

    Usage: Y = leaky_relu_map_sym(S, 0.1, sgn)
    """
    return 0.5 * ((1.0 + a) * M + (1.0 - a) * matrix_abs(M, sgn))


def capped_leaky_relu_map_sym(M: torch.Tensor, a: float, beta: float, sgn: msgn) -> torch.Tensor:
    """(1/2) [beta I + a M + (1 - a) |M| - |M - beta I|].

    Usage: Y = capped_leaky_relu_map_sym(S, 0.1, 0.5, sgn)
    """
    I = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    return 0.5 * (beta * I + a * M + (1.0 - a) * matrix_abs(M, sgn) - matrix_abs(M - beta * I, sgn))


def leaky_clip_map_sym(M: torch.Tensor, a: float, mu: float, sgn: msgn) -> torch.Tensor:
    """a M + (1 - a)/2 [|M + mu I| - |M - mu I|].

    Usage: Y = leaky_clip_map_sym(S, 0.1, 0.5, sgn)
    """
    I = torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
    return a * M + 0.5 * (1.0 - a) * (matrix_abs(M + mu * I, sgn) - matrix_abs(M - mu * I, sgn))
