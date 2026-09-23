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

U0 = 0.6
N_ITERS_SCALAR = 8

SIGMA_MIN = 0.6
MATRIX_SIZE = 30
N_ITERS_MATRIX = 6


def _errors(log10_err: torch.Tensor) -> list[float]:
    return (10.0**log10_err).tolist()


def main() -> None:
    generator = torch.Generator().manual_seed(0)

    scalar = convergence_order.degree_sweep_scalar(D_VALUES, U0, N_ITERS_SCALAR)
    matrix = convergence_order.degree_sweep_matrix(
        D_VALUES, SIGMA_MIN, MATRIX_SIZE, N_ITERS_MATRIX, generator=generator
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plotting.plot_error_decay(
        axes[0],
        list(range(N_ITERS_SCALAR + 1)),
        {f"D={D}": _errors(scalar[D]["log10_error"]) for D in D_VALUES},
        title=f"scalar orbit error |1 - u_k|, u0={U0}",
        ylabel="error (deflated past 1e-8)",
    )
    plotting.plot_convergence_order(
        axes[1],
        {f"D={D}": scalar[D]["order_estimate"] for D in D_VALUES},
        {f"D={D}": scalar[D]["predicted_order"] for D in D_VALUES},
        title="empirical order estimate (scalar orbit)",
    )
    plotting.save_figure(fig, OUTDIR, "convergence_order_scalar.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plotting.plot_error_decay(
        axes[0],
        list(range(N_ITERS_MATRIX + 1)),
        {f"D={D}": _errors(matrix[D]["matrix_log10_error"]) for D in D_VALUES},
        title=f"matrix NS iteration error, sigma_min={SIGMA_MIN}, n={MATRIX_SIZE}",
        ylabel=r"$\|X_k - \mathrm{Sgn}(M)\|_2$",
    )
    plotting.plot_convergence_order(
        axes[1],
        {f"D={D}": matrix[D]["order_estimate"] for D in D_VALUES},
        {f"D={D}": matrix[D]["predicted_order"] for D in D_VALUES},
        title="empirical order estimate (matrix iteration)",
    )
    plotting.save_figure(fig, OUTDIR, "convergence_order_matrix.png")

    floor = matrix[D_VALUES[0]]["precision_floor"]
    print(
        f"matrix-level errors are only meaningful above the float64 "
        f"precision floor (~1e{floor:.0f}); order estimates computed from "
        f"errors below that are noise, not a violation of the theory."
    )


if __name__ == "__main__":
    main()
