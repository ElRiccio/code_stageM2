"""Reproducible driver for the convergence-order validation figures
(reviewer comment on verifying the predicted order D+1, not just observing
convergence "to machine precision").

Run as `python -m experiments.scripts.convergence_order [--device {auto,cpu,cuda}] [--dtype {float32,float64}]`.
`--device`/`--dtype` affect only the matrix-iteration check (the scalar
orbit is plain-float64 Python arithmetic, unaffected by either).
"""

from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import torch

from experiments import cli, convergence_order, plotting

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "figures")

D_VALUES = [1, 2, 3, 4]

U0 = 0.1
N_ITERS_SCALAR = 6
# deflate=False: plot the scalar orbit's error by plain subtraction, the
# same operation the matrix case is stuck with, so it hits the same
# float64 noise floor after a few steps instead of the deflated form's
# unlimited decades. See ns_iteration.log10_error_orbit's docstring.
DEFLATE_SCALAR = False

# A small sigma_min (matching U0) rather than a comfortable 0.6: the whole
# point of order D+1 is that it converges explosively fast, so a starting
# point close to 1 burns through the entire pre-floor transient in 1-2
# steps at high D, leaving almost nothing to fit a slope to. Starting
# further from 1 stretches that transient out, at the cost of needing a
# couple more iterations to reach it.
SIGMA_MIN = 0.1
MATRIX_M, MATRIX_N = 40, 25  # rectangular on purpose; nothing here needs square
N_ITERS_MATRIX = 8


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
        p = results[D]["predicted_order"]
        log10_const = math.log10(results[D]["asymptotic_constant"])
        plotting.plot_loglog_order(
            ax,
            {f"D={D}": log10_err},
            {f"D={D}": (p, log10_const)},
            title=f"D={D} (slope {p})",
            ylabel=r"$\log_{10} e_{k+1}$" if D == D_VALUES[0] else "",
        )
    fig.suptitle(suptitle)
    return fig


def main() -> None:
    device, dtype = cli.parse_device_dtype(
        default_dtype=torch.float64, description="Empirical order-(D+1) convergence validation."
    )
    generator = torch.Generator(device=device).manual_seed(0)

    scalar = convergence_order.degree_sweep_scalar(
        D_VALUES, U0, N_ITERS_SCALAR, deflate=DEFLATE_SCALAR
    )
    matrix = convergence_order.degree_sweep_matrix(
        D_VALUES, SIGMA_MIN, MATRIX_M, MATRIX_N, N_ITERS_MATRIX,
        generator=generator, device=device, dtype=dtype,
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    plotting.plot_error_decay(
        ax,
        list(range(N_ITERS_SCALAR + 1)),
        {f"D={D}": _errors(scalar[D]["log10_error"]) for D in D_VALUES},
        title=f"scalar orbit error |1 - u_k|, u0={U0}"
        + (" (raw, no deflation)" if not DEFLATE_SCALAR else ""),
        ylabel="error",
    )
    plotting.save_figure(fig, OUTDIR, "convergence_order_scalar_error.png")

    fig = _loglog_grid(scalar, "log10_error", "order = slope on the log-log plot (scalar orbit)")
    plotting.save_figure(fig, OUTDIR, "convergence_order_scalar_loglog.png")

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    plotting.plot_error_decay(
        ax,
        list(range(N_ITERS_MATRIX + 1)),
        {f"D={D}": _errors(matrix[D]["matrix_log10_error"]) for D in D_VALUES},
        title=f"matrix NS iteration error, sigma_min={SIGMA_MIN}, {MATRIX_M}x{MATRIX_N}, "
        f"{device}/{cli.dtype_name(dtype)}",
        ylabel=r"$\|X_k - \mathrm{Sgn}(M)\|_2$",
    )
    plotting.save_figure(fig, OUTDIR, f"convergence_order_matrix_error_{device.type}.png")

    fig = _loglog_grid(
        matrix, "matrix_log10_error",
        f"order = slope on the log-log plot (matrix iteration, {device}/{cli.dtype_name(dtype)})",
    )
    plotting.save_figure(fig, OUTDIR, f"convergence_order_matrix_loglog_{device.type}.png")

    floor = matrix[D_VALUES[0]]["precision_floor"]
    print(
        f"matrix-level errors are only meaningful above the {cli.dtype_name(dtype)} "
        f"precision floor (~1e{floor:.0f}); order estimates computed from "
        f"errors below that are noise, not a violation of the theory."
    )


if __name__ == "__main__":
    main()