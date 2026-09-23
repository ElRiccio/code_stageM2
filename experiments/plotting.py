"""Shared plotting helpers, kept deliberately generic: every function here
takes plain arrays/tensors and axis labels, never an experiment's own result
object, so the same helper serves several experiments.

Figures are saved as plain PNG/PDF files via matplotlib; no LaTeX-ready
(pgfplots/tikz) output is produced.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return torch.as_tensor(t).detach().cpu().numpy()


def setup_axis(ax, xlabel: str, ylabel: str, title: str, legend: bool = True) -> None:
    """Apply the shared axis conventions: labels, title, light grid, legend."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    if legend:
        ax.legend(fontsize=8)


def plot_spectrum(
    ax,
    sigma: torch.Tensor,
    exact: torch.Tensor,
    approx_by_k: dict[int, torch.Tensor],
    title: str,
    xaxis: str = "index",
    ylabel: str = "spectral coordinate",
) -> None:
    """Input spectrum (dotted), exact target (dashed), and the approximate
    spectral coordinates at a handful of NS budgets k (solid).

    `xaxis` is "index" (points ordered, plotted against 1..len(sigma)) or
    "sigma" (plotted against the singular/eigen values themselves).
    """
    sigma_np = _to_numpy(sigma)
    order = np.argsort(sigma_np)
    s = sigma_np[order]
    if xaxis == "index":
        x, xlabel = np.arange(1, s.size + 1), "index i (increasing sigma)"
    elif xaxis == "sigma":
        x, xlabel = s, r"$\sigma_i(M)$"
    else:
        raise ValueError("xaxis must be 'index' or 'sigma'")

    ax.plot(x, s, ":", color="0.35", lw=1.2, label=r"$\sigma_i(M)$")
    for k, v in approx_by_k.items():
        ax.plot(x, _to_numpy(v)[order], lw=1.0, label="k=%d" % int(k))
    ax.plot(x, _to_numpy(exact)[order], "k--", lw=1.3, label="exact (SVD)")
    setup_axis(ax, xlabel, ylabel, title)


def plot_error_decay(
    ax,
    k_values: list[int],
    error_tracks: dict[str, list[float]],
    title: str,
    xlabel: str = "NS steps k",
    ylabel: str = "error",
    floor: float = 1e-18,
) -> None:
    """Semilog decay of one or several named error tracks against NS budget
    k, floored away from exactly zero so a log axis stays well defined."""
    for label, errs in error_tracks.items():
        errs_np = np.maximum(np.asarray(errs, dtype=float), floor)
        ax.semilogy(k_values, errs_np, "o-", ms=3, lw=1.0, label=label)
    setup_axis(ax, xlabel, ylabel, title)


def plot_convergence_order(
    ax,
    order_tracks: dict[str, torch.Tensor],
    predicted_orders: dict[str, float] | None = None,
    title: str = "",
    xlabel: str = "iteration k",
    ylabel: str = "empirical order estimate",
) -> None:
    """Empirical order estimate p_hat_k against iteration k for one or
    several named tracks (e.g. one per degree D), each drawn against its own
    predicted order as a dashed horizontal line in the same color, when
    `predicted_orders` gives one for that label."""
    predicted_orders = predicted_orders or {}
    for i, (label, orders) in enumerate(order_tracks.items()):
        color = "C%d" % i
        orders_np = _to_numpy(orders)
        k = np.arange(orders_np.size)
        ax.plot(k, orders_np, "o-", ms=3, lw=1.0, color=color, label=label)
        if label in predicted_orders:
            ax.axhline(predicted_orders[label], ls="--", lw=0.8, color=color)
    setup_axis(ax, xlabel, ylabel, title)


def save_figure(fig, outdir: str, name: str, dpi: int = 150, show: bool = False) -> None:
    """Tight-layout, save `fig` to outdir/name, and either display or close it."""
    fig.tight_layout()
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, name), dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
