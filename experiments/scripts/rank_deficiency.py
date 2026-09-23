"""Reproducible driver for the rank-deficiency and conditioning sweep
figure (reviewer comments: several ranks and several smallest nonzero
singular values; the influence of the polynomial degree D should also be
studied).

Single random matrix per (D, swept value) for now - no repeats/averaging
yet; that's a natural follow-up once this shape is settled.

Run as `python -m experiments.scripts.rank_deficiency`.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import torch

from experiments import plotting, rank_deficiency

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "figures")

M, N = 40, 25  # rectangular on purpose, as elsewhere in the project
R = min(M, N)
D_VALUES = [1, 2, 3, 4]
N_ITERS = 60  # generous: D=1 (order-2 convergence) needs far more steps than D=4
TOL = 1e-10

N_ZERO_VALUES = [0, R // 4, R // 2, 3 * R // 4, R - 1]
SIGMA_MIN_VALUES = [0.5, 0.1, 0.01, 1e-3, 1e-4]


def _as_plot_series(iters_by_x: dict, x_values: list[float]) -> list[float]:
    """iterations, with unconverged points (-1) mapped to NaN so the line
    breaks there instead of plotting a misleading -1."""
    return [float("nan") if iters_by_x[x] < 0 else iters_by_x[x] for x in x_values]


def main() -> None:
    generator = torch.Generator().manual_seed(0)

    rank_results = rank_deficiency.rank_sweep(
        N_ZERO_VALUES, M, N, D_VALUES, N_ITERS, tol=TOL, generator=generator,
    )
    cond_results = rank_deficiency.conditioning_sweep(
        SIGMA_MIN_VALUES, M, N, D_VALUES, N_ITERS, tol=TOL, generator=generator,
    )

    ranks = [R - nz for nz in N_ZERO_VALUES]

    fig, (ax_rank, ax_cond) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for i, D in enumerate(D_VALUES):
        color = "C%d" % i
        plotting.plot_metric_vs_parameter(
            ax_rank,
            ranks,
            _as_plot_series(rank_results[D], N_ZERO_VALUES),
            xlabel=f"rank (out of {R})",
            ylabel=f"NS iterations to reach error <= {TOL:.0e}",
            title="vs. rank",
            label=f"D={D}",
            color=color,
        )
        plotting.plot_metric_vs_parameter(
            ax_cond,
            SIGMA_MIN_VALUES,
            _as_plot_series(cond_results[D], SIGMA_MIN_VALUES),
            xlabel=r"$\sigma_{\min}$",
            ylabel="",
            title="vs. $\\sigma_{\\min}$",
            xscale="log",
            label=f"D={D}",
            color=color,
        )
    fig.suptitle(f"iterations to tolerance, {M}x{N} (single matrix per point, no averaging yet)")
    plotting.save_figure(fig, OUTDIR, "rank_deficiency_iterations.png")

    for label, results, x_values in [
        ("rank sweep", rank_results, N_ZERO_VALUES),
        ("conditioning sweep", cond_results, SIGMA_MIN_VALUES),
    ]:
        for D in D_VALUES:
            not_converged = [x for x in x_values if results[D][x] < 0]
            if not_converged:
                print(
                    f"{label}, D={D}: did not reach tol={TOL:.0e} within {N_ITERS} "
                    f"iterations at {not_converged} (excluded from the plot, not clipped)."
                )


if __name__ == "__main__":
    main()