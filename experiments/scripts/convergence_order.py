"""Reproducible driver for the convergence-order validation figures
(reviewer comment on verifying the predicted order D+1, not just observing
convergence "to machine precision").

Run as `python -m experiments.scripts.convergence_order`.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import torch

from experiments import convergence_order, plotting

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "figures")

D_VALUES = [1, 2, 3, 4]

U0 = 0.1
N_ITERS_SCALAR = 6

SIGMA_MIN = 0.6
MATRIX_SIZE = 30
N_ITERS_MATRIX = 6


def _errors(log10_err: torch.Tensor) -> list[float]:
    return (10.0**log10_err).tolist()


def _drop_after_stall(
    log10_err: torch.Tensor, min_decades_drop: float = 0.5, deep_threshold: float = -10.0
) -> torch.Tensor:
    """Drop entries once the error, already deep (below `deep_threshold`,
    comfortably past any slow-starting transient), stops decreasing by at
    least `min_decades_drop` per step: that combination is where a
    finite-precision computation (the matrix iteration) has hit its
    floating-point noise floor. Past that point, further "iterates" are
    noise clustered near the floor, not data on the order-(D+1) line, so
    plotting them on the log-log diagnostic would misrepresent them as
    measurements. Requiring `deep_threshold` first avoids mistaking a
    genuinely slow early step (e.g. D=1's first step, order 2 has not yet
    kicked in) for a floor. A no-op for a sequence (like the deflated
    scalar orbit) that never stalls.
    """
    if log10_err.numel() < 2:
        return log10_err
    diffs = log10_err[1:] - log10_err[:-1]
    deep_enough = log10_err[:-1] < deep_threshold
    stalled = ((diffs > -min_decades_drop) & deep_enough).nonzero(as_tuple=True)[0]
    if stalled.numel() == 0:
        return log10_err
    return log10_err[: int(stalled[0]) + 1]


def _loglog_grid(results: dict[int, dict], log10_key: str, suptitle: str) -> plt.Figure:
    """One log-log-slope panel per D, each on its own axis scale: the D+1
    order sequences blow up at wildly different rates (order 5 reaches
    log10 error ~ -100 in a couple of steps, order 2 barely moves), so a
    single shared axis makes the small-D tracks collapse to a point next to
    the large-D ones. Small multiples keep every degree legible.
    """
    fig, axes = plt.subplots(1, len(D_VALUES), figsize=(3.2 * len(D_VALUES), 3.6))
    for ax, D in zip(axes, D_VALUES):
        log10_err = _drop_after_stall(results[D][log10_key])
        plotting.plot_loglog_order(
            ax,
            {f"D={D}": log10_err},
            {f"D={D}": results[D]["predicted_order"]},
            title=f"D={D} (slope {results[D]['predicted_order']})",
            ylabel=r"$\log_{10} e_{k+1}$" if D == D_VALUES[0] else "",
        )
    fig.suptitle(suptitle)
    return fig


def main() -> None:
    generator = torch.Generator().manual_seed(0)

    scalar = convergence_order.degree_sweep_scalar(D_VALUES, U0, N_ITERS_SCALAR)
    matrix = convergence_order.degree_sweep_matrix(
        D_VALUES, SIGMA_MIN, MATRIX_SIZE, N_ITERS_MATRIX, generator=generator
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    plotting.plot_error_decay(
        ax,
        list(range(N_ITERS_SCALAR + 1)),
        {f"D={D}": _errors(scalar[D]["log10_error"]) for D in D_VALUES},
        title=f"scalar orbit error |1 - u_k|, u0={U0}",
        ylabel="error (deflated past 1e-8)",
    )
    plotting.save_figure(fig, OUTDIR, "convergence_order_scalar_error.png")

    fig = _loglog_grid(scalar, "log10_error", "order = slope on the log-log plot (scalar orbit)")
    plotting.save_figure(fig, OUTDIR, "convergence_order_scalar_loglog.png")

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    plotting.plot_error_decay(
        ax,
        list(range(N_ITERS_MATRIX + 1)),
        {f"D={D}": _errors(matrix[D]["matrix_log10_error"]) for D in D_VALUES},
        title=f"matrix NS iteration error, sigma_min={SIGMA_MIN}, n={MATRIX_SIZE}",
        ylabel=r"$\|X_k - \mathrm{Sgn}(M)\|_2$",
    )
    plotting.save_figure(fig, OUTDIR, "convergence_order_matrix_error.png")

    fig = _loglog_grid(
        matrix, "matrix_log10_error", "order = slope on the log-log plot (matrix iteration)"
    )
    plotting.save_figure(fig, OUTDIR, "convergence_order_matrix_loglog.png")

    floor = matrix[D_VALUES[0]]["precision_floor"]
    print(
        f"matrix-level errors are only meaningful above the float64 "
        f"precision floor (~1e{floor:.0f}); order estimates computed from "
        f"errors below that are noise, not a violation of the theory."
    )


if __name__ == "__main__":
    main()
