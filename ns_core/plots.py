"""Matplotlib helpers for the convergence experiments. Each takes results
keyed by the degree D (a dict D -> tensor over the iterations) and an
optional axis, draws on it (a new one if omitted) and returns it. Degrees
keep the same colour in every plot.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D

from ns_core import ns_iteration, orbit_tools


def _axis(ax):
    return plt.subplots(figsize=(5.5, 4))[1] if ax is None else ax


def _np(t: torch.Tensor):
    return t.detach().cpu().numpy()


def _colors(degrees) -> dict:
    cycle = plt.get_cmap("tab10")
    return {D: cycle(i % 10) for i, D in enumerate(sorted(degrees))}


def plot_error_curves(
    errors: dict[int, torch.Tensor],
    ax=None,
    eps: float | None = None,
    predicted: dict[int, int] | None = None,
):
    """Error against iteration k on a log scale, one curve per D. With `eps`,
    draws that level and marks the first k at or below it; with `predicted`
    (D -> K_D), draws a dashed vertical line at each predicted count.

    Usage: plot_error_curves(res["error"], eps=1e-6, predicted=res["predicted"])
    """
    ax = _axis(ax)
    colors = _colors(errors)
    for D in sorted(errors):
        e = errors[D]
        ax.semilogy(_np(torch.arange(len(e))), _np(e), "-o", ms=3, color=colors[D])
        if predicted is not None and D in predicted:
            ax.axvline(predicted[D], color=colors[D], ls="--", lw=1)
        if eps is not None:
            k = orbit_tools.first_hit(e, eps)
            if k is not None:
                ax.plot(k, float(e[k]), "s", ms=9, mfc="none", color=colors[D])
    handles = [Line2D([], [], color=colors[D], marker="o", ms=3, label=f"D={D}") for D in sorted(errors)]
    if eps is not None:
        ax.axhline(eps, color="gray", ls=":")
        handles.append(Line2D([], [], color="k", marker="s", mfc="none", ls="", label="first k with error $\\leq \\varepsilon$"))
    if predicted is not None:
        handles.append(Line2D([], [], color="k", ls="--", lw=1, label="predicted $K_D$"))
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel("$\\|X_k - \\mathrm{Sgn}(M)\\|_2$")
    ax.legend(handles=handles, fontsize=8)
    return ax


def plot_error_map(errors: dict[int, torch.Tensor], ax=None, floor: float | None = None):
    """Log-log plot of e_{k+1} against e_k, one point cloud per D, with the
    asymptotic law e_{k+1} = kappa_D e_k^(D+1) dashed in the same colour
    (slope D+1). Points with e_{k+1} at or below `floor` are dropped; by
    default the floor is 3x the final error of a curve that has stagnated
    (last two errors within a factor 2), and 0 otherwise.

    Usage: plot_error_map(res["error"])
    """
    ax = _axis(ax)
    colors = _colors(errors)
    for D in sorted(errors):
        e = errors[D]
        x, y = e[:-1], e[1:]
        if floor is not None:
            level = floor
        else:
            stagnated = len(e) > 1 and float(e[-1]) >= 0.5 * float(e[-2])
            level = 3.0 * float(e[-1]) if stagnated else 0.0
        keep = y > level
        x, y = x[keep], y[keep]
        ax.loglog(_np(x), _np(y), "o", ms=4, color=colors[D], label=f"D={D}")
        if x.numel():
            line = torch.logspace(torch.log10(x.min()), torch.log10(x.max()), 50, dtype=x.dtype)
            kappa = ns_iteration.asymptotic_error_constant(D)
            ax.loglog(_np(line), _np(kappa * line ** (D + 1)), "--", lw=1, color=colors[D])
    ax.set_xlabel("$e_k$")
    ax.set_ylabel("$e_{k+1}$")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="k", ls="--", lw=1, label="$\\kappa_D\\, e_k^{D+1}$"))
    ax.legend(handles=handles, fontsize=8)
    return ax


def plot_ranks(ranks: dict[int, torch.Tensor], r: int | None = None, ax=None):
    """Numerical rank of X_k against iteration k, one curve per D; with `r`,
    draws the rank of M as a dashed horizontal line.

    Usage: plot_ranks(res["rank"], r=orbit_tools.resolve_rank(m, n, rank))
    """
    ax = _axis(ax)
    colors = _colors(ranks)
    for D in sorted(ranks):
        k = _np(torch.arange(len(ranks[D])))
        ax.plot(k, _np(ranks[D]), "-o", ms=3, color=colors[D], label=f"D={D}")
    if r is not None:
        ax.axhline(r, color="gray", ls="--", label=f"rank of $M$ = {r}")
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel("numerical rank of $X_k$")
    ax.legend(fontsize=8)
    return ax
