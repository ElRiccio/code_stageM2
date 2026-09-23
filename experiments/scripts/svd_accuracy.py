"""Reproducible driver for the accuracy-vs-SVD table/figure (reviewer:
"a quantitative table would be useful", reporting relative Frobenius error
per operator).

Run as `python -m experiments.scripts.svd_accuracy [--device {auto,cpu,cuda}] [--dtype {float32,float64}]`.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments import cli, map_catalogue, svd_accuracy, tables

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESDIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    device, dtype = cli.parse_device_dtype(
        default_dtype=torch.float64, description="Accuracy vs. exact SVD, per map."
    )
    map_specs = map_catalogue.all_maps()
    results = svd_accuracy.accuracy_experiment(
        map_specs,
        svd_accuracy.DEFAULT_M,
        svd_accuracy.DEFAULT_N,
        svd_accuracy.DEFAULT_D,
        device=device,
        dtype=dtype,
    )

    rows = [
        {
            "map": name,
            "device": str(device),
            "dtype": cli.dtype_name(dtype),
            "m": svd_accuracy.DEFAULT_M,
            "n": svd_accuracy.DEFAULT_N,
            "D": svd_accuracy.DEFAULT_D,
            "n_trials": len(summary.values),
            "mean_relative_frobenius_error": summary.mean,
            "std_relative_frobenius_error": summary.std,
            "seeds": " ".join(str(s) for s in summary.seeds),
        }
        for name, summary in results.items()
    ]
    csv_path = tables.save_csv(rows, RESDIR, f"accuracy_table_{device.type}.csv")

    names = [r["map"] for r in rows]
    means = np.array([r["mean_relative_frobenius_error"] for r in rows])
    stds = np.array([r["std_relative_frobenius_error"] for r in rows])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=3, color="C0")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(r"relative Frobenius error $\|\hat\Phi(M)-\Phi(M)\|_F / \|\Phi(M)\|_F$")
    ax.set_title(
        f"Accuracy vs. exact SVD, {svd_accuracy.DEFAULT_M}x{svd_accuracy.DEFAULT_N}, "
        f"D={svd_accuracy.DEFAULT_D}, {device}/{cli.dtype_name(dtype)} "
        f"({len(rows[0]['seeds'].split())} trials, mean +/- std)"
    )
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig_path = os.path.join(FIGDIR, f"svd_accuracy_{device.type}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {fig_path}")
    for r in rows:
        print(f"  {r['map']:20s} {r['mean_relative_frobenius_error']:.3e} +/- {r['std_relative_frobenius_error']:.3e}")


if __name__ == "__main__":
    main()
