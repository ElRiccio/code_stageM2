"""Reproducible driver for the accuracy-vs-SVD table/figure (reviewer:
"a quantitative table would be useful", reporting relative Frobenius error
per operator).

Run as `python -m experiments.scripts.svd_accuracy`.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from experiments import map_catalogue, svd_accuracy, tables

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESDIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    map_specs = map_catalogue.all_maps()
    results = svd_accuracy.accuracy_experiment(
        map_specs,
        svd_accuracy.DEFAULT_M,
        svd_accuracy.DEFAULT_N,
        svd_accuracy.DEFAULT_D,
    )

    rows = [
        {
            "map": name,
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
    csv_path = tables.save_csv(rows, RESDIR, "accuracy_table.csv")

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
        f"D={svd_accuracy.DEFAULT_D} ({len(rows[0]['seeds'].split())} trials, mean +/- std)"
    )
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig_path = os.path.join(FIGDIR, "svd_accuracy.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {fig_path}")
    for r in rows:
        print(f"  {r['map']:20s} {r['mean_relative_frobenius_error']:.3e} +/- {r['std_relative_frobenius_error']:.3e}")


if __name__ == "__main__":
    main()
