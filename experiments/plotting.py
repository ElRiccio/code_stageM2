"""Shared plotting helpers, kept deliberately generic: every function here
takes plain arrays/tensors and axis labels, never an experiment's own result
object, so the same helper serves several experiments.

Figures are saved as plain PNG/PDF files via matplotlib; no LaTeX-ready
(pgfplots/tikz) output is produced.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import torch


def setup_axis(ax, xlabel: str, ylabel: str, title: str, legend: bool = True) -> None:
    """Apply the shared axis conventions: labels, title, light grid, legend."""
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def save_figure(fig, outdir: str, name: str, dpi: int = 150, show: bool = False) -> None:
    """Tight-layout, save `fig` to outdir/name, and either display or close it."""
    raise NotImplementedError
