"""The catalogue of maps under test in the SVD-comparison experiment suite
(reviewer comments: "there should be a direct comparison with the SVD-based
implementation"; "the six activation/profile functions ... should be
compared systematically").

Five of the six CPWL profiles of ns_core.cpwl (all but the leaky rectifier,
excluded here since its rectangular sign form is literally the identity —
ns_core.cpwl.leaky_relu_map's docstring: "the sign form does not see the
leak" — so an SVD-ground-truth comparison for it is vacuous, matching the
thesis's own cor:no-leak), plus the matrix sign map itself (`msgn`): not one
of the six CPWL profiles, but the primitive they are all built from, and
still a meaningful decomposition-free-vs-SVD comparison in its own right
(ns_core.sign_map.make_sgn_ns vs. ns_core.sign_map.sgn_svd).

This module is deliberately not part of ns_core: the parameter choices below
(alpha, beta, gamma, mu, a) are specific to this experiment suite's figures,
the way legacy_numpy/exp13_catalogue.py's ALPHA/BETA/GAMMA/... constants were
specific to its own figures, not reusable library content. It only composes
existing ns_core.cpwl / ns_core.sign_map / ns_core.metrics exports; nothing
here reimplements a decomposition-free evaluator or a reference operator.

Every MapSpec exposes:
  - `evaluate(M, sgn)`: the decomposition-free result, a function of M and a
    Sgn callable (exact or NS surrogate; see ns_core.sign_map).
  - `reference(M)`: the exact SVD-based ground truth, computed independently
    of `evaluate` (via ns_core.metrics.op_svd + the matching scalar profile,
    or ns_core.sign_map.sgn_svd directly for `msgn`).
All maps act on R^{m x n} (the rectangular sign forms), since the SVD is the
ground truth requested by the reviewer; the symmetric/eigenvalue forms are
out of scope for this suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ns_core import cpwl, matrices, metrics, sign_map

Sgn = Callable[[torch.Tensor], torch.Tensor]


def unit_norm_gaussian(
    m: int, n: int, *, generator: torch.Generator, device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """A Gaussian m x n matrix rescaled to exact spectral norm 1: the
    normalization every knot in this catalogue (CLIP_ALPHA, SOFT_GAMMA, ...)
    is chosen against, and the convention shared by every axis module in
    this suite that draws a plain (not rank/conditioning-controlled)
    Gaussian test matrix."""
    M = matrices.rand_gaussian(m, n, generator=generator, device=device, dtype=dtype)
    beta = sign_map.spectral_norm_exact(M)
    return M / beta if beta > 0 else M


@dataclass(frozen=True)
class MapSpec:
    """One map under test: its name, the parameters it was built with (for
    labeling), the decomposition-free evaluator, and the exact SVD-based
    reference."""

    name: str
    params: dict
    evaluate: Callable[[torch.Tensor, Sgn], torch.Tensor]
    reference: Callable[[torch.Tensor], torch.Tensor]


# Parameters read against ||M||_2 = 1 (every matrix generator in this suite
# normalizes to unit spectral norm), so every knot below sits inside the
# spectrum. Chosen to match legacy_numpy/exp13_catalogue.py's constants for
# continuity with the earlier NumPy figures (ALPHA/BETA, GAMMA, A_LEAK,
# BETA_C, MU_LKC); MU_HINGE has no legacy precedent and is chosen in the
# same spirit (comfortably inside (0, 1)).
CLIP_ALPHA, CLIP_BETA = -0.30, 0.60
SOFT_GAMMA = 0.25
MU_HINGE = 0.30
LEAK_A = 0.20
CAP_BETA = 0.55
LKC_A, LKC_MU = 0.20, 0.40


def _clip_map_spec() -> MapSpec:
    alpha, beta = CLIP_ALPHA, CLIP_BETA

    def evaluate(M: torch.Tensor, sgn: Sgn) -> torch.Tensor:
        return cpwl.clip_map(M, alpha, beta, sgn)

    scalar = cpwl.odd_extension(lambda x: cpwl.clip(x, alpha, beta))

    def reference(M: torch.Tensor) -> torch.Tensor:
        return metrics.op_svd(M, scalar)

    return MapSpec("clip", {"alpha": alpha, "beta": beta}, evaluate, reference)


def _soft_map_spec() -> MapSpec:
    gamma = SOFT_GAMMA

    def evaluate(M: torch.Tensor, sgn: Sgn) -> torch.Tensor:
        return cpwl.soft_map(M, gamma, sgn)

    # soft_threshold is odd, hence its own odd extension (Table 6.1: odd_h = h).
    def reference(M: torch.Tensor) -> torch.Tensor:
        return metrics.op_svd(M, lambda x: cpwl.soft_threshold(x, gamma))

    return MapSpec("soft", {"gamma": gamma}, evaluate, reference)


def _hinge_map_spec() -> MapSpec:
    mu = MU_HINGE

    def evaluate(M: torch.Tensor, sgn: Sgn) -> torch.Tensor:
        return cpwl.hinge_map(M, mu, sgn)

    # hinge_mu(x) = ReLU(x - mu); cpwl.py has no standalone scalar hinge, so
    # it is composed here from the exported relu, exactly as the odd
    # extension of ReLU(. - mu) that Table 6.1 identifies as soft_mu.
    scalar = cpwl.odd_extension(lambda x: cpwl.relu(x - mu))

    def reference(M: torch.Tensor) -> torch.Tensor:
        return metrics.op_svd(M, scalar)

    return MapSpec("hinge", {"mu": mu}, evaluate, reference)


def _capped_leaky_relu_map_spec() -> MapSpec:
    a, beta = LEAK_A, CAP_BETA

    def evaluate(M: torch.Tensor, sgn: Sgn) -> torch.Tensor:
        return cpwl.capped_leaky_relu_map(M, a, beta, sgn)

    scalar = cpwl.odd_extension(lambda x: cpwl.capped_leaky_relu(x, a, beta))

    def reference(M: torch.Tensor) -> torch.Tensor:
        return metrics.op_svd(M, scalar)

    return MapSpec("capped_leaky_relu", {"a": a, "beta": beta}, evaluate, reference)


def _leaky_clip_map_spec() -> MapSpec:
    a, mu = LKC_A, LKC_MU

    def evaluate(M: torch.Tensor, sgn: Sgn) -> torch.Tensor:
        return cpwl.leaky_clip_map(M, a, mu, sgn)

    # leaky_clip is odd by construction (cpwl.leaky_clip's docstring), hence
    # its own odd extension.
    def reference(M: torch.Tensor) -> torch.Tensor:
        return metrics.op_svd(M, lambda x: cpwl.leaky_clip(x, a, mu))

    return MapSpec("leaky_clip", {"a": a, "mu": mu}, evaluate, reference)


def _msgn_map_spec() -> MapSpec:
    # Not a CPWL profile: the matrix sign map itself. `evaluate` ignores M
    # and applies the given Sgn surrogate directly; `reference` is the exact
    # sign_map.sgn_svd, independent of whichever Sgn surrogate is under test.
    def evaluate(M: torch.Tensor, sgn: Sgn) -> torch.Tensor:
        return sgn(M)

    def reference(M: torch.Tensor) -> torch.Tensor:
        return sign_map.sgn_svd(M)

    return MapSpec("msgn", {}, evaluate, reference)


def all_maps() -> list[MapSpec]:
    """The six maps under test, in a fixed order used throughout this suite's
    tables and plots."""
    return [
        _clip_map_spec(),
        _soft_map_spec(),
        _hinge_map_spec(),
        _capped_leaky_relu_map_spec(),
        _leaky_clip_map_spec(),
        _msgn_map_spec(),
    ]
