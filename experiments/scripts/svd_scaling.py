"""Reproducible driver for the scaling-with-matrix-size figure (reviewer:
"a scaling experiment with increasing m, n would be particularly
important").

Run as `python -m experiments.scripts.svd_scaling`.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from experiments import map_catalogue, plotting, svd_scaling, tables

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESDIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    map_specs = map_catalogue.all_maps()
    results = svd_scaling.scaling_experiment(map_specs)

    rows = []
    for name, by_size in results.items():
        for (m, n), quantities in by_size.items():
            err, free, svd = quantities["error"], quantities["decomposition_free_s"], quantities["svd_reference_s"]
            rows.append({
                "map": name, "m": m, "n": n, "D": svd_scaling.DEFAULT_D,
                "n_trials": len(err.values),
                "mean_relative_frobenius_error": err.mean, "std_relative_frobenius_error": err.std,
                "mean_decomposition_free_s": free.mean, "std_decomposition_free_s": free.std,
                "mean_svd_reference_s": svd.mean, "std_svd_reference_s": svd.std,
                "seeds": " ".join(str(s) for s in err.seeds),
            })
    csv_path = tables.save_csv(rows, RESDIR, "scaling_table.csv")

    sizes = svd_scaling.SIZES
    ns = np.array([n for _, n in sizes])

    fig, (ax_err, ax_time) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for i, spec in enumerate(map_specs):
        color = f"C{i}"
        by_size = results[spec.name]
        err_means = [by_size[s]["error"].mean for s in sizes]
        err_stds = [by_size[s]["error"].std for s in sizes]
        plotting.plot_metric_vs_parameter(
            ax_err, ns.tolist(), err_means, std=err_stds,
            xlabel="n (matrix size, m:n = 4:3)", ylabel="relative Frobenius error",
            title="accuracy vs. size", xscale="log", label=spec.name, color=color,
        )
        free_means = [by_size[s]["decomposition_free_s"].mean for s in sizes]
        svd_means = [by_size[s]["svd_reference_s"].mean for s in sizes]
        ax_time.loglog(ns, free_means, "o-", ms=4, color=color, label=f"{spec.name} (NS)")
        ax_time.loglog(ns, svd_means, "s--", ms=4, color=color, alpha=0.5)
    ax_err.set_yscale("log")
    ax_time.set_xlabel("n (matrix size, m:n = 4:3)")
    ax_time.set_ylabel("wall-clock time (s, min of repeats)")
    ax_time.set_title("time vs. size (solid=NS, dashed=SVD)")
    ax_time.grid(True, which="both", linewidth=0.3, alpha=0.5)
    ax_time.legend(fontsize=6, ncol=2)
    fig.suptitle("Scaling with matrix size (fewer trials at larger sizes; see CSV for n_trials per point)")
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig_path = os.path.join(FIGDIR, "svd_scaling.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
