"""Reproducible driver for the degree-D comparison figure of the SVD-
comparison suite (reviewer: "the matrix experiments use D=3 throughout ...
compare D=1,2,3,... numerically and verify the conclusions of Chapter 4").

Run as `python -m experiments.scripts.svd_degree [--device {auto,cpu,cuda}] [--dtype {float32,float64}]`.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import torch

from experiments import cli, map_catalogue, plotting, svd_degree, tables

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESDIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    device, dtype = cli.parse_device_dtype(
        default_dtype=torch.float64, description="Dependence on degree D."
    )
    map_specs = map_catalogue.all_maps()
    results = svd_degree.degree_sweep(map_specs, device=device, dtype=dtype)

    rows = []
    for map_name, by_D in results.items():
        for D, quantities in by_D.items():
            err, free = quantities["error"], quantities["decomposition_free_s"]
            rows.append({
                "map": map_name, "D": D, "device": str(device), "dtype": cli.dtype_name(dtype),
                "n_trials": len(err.values),
                "mean_relative_frobenius_error": err.mean, "std_relative_frobenius_error": err.std,
                "mean_decomposition_free_s": free.mean, "std_decomposition_free_s": free.std,
                "seeds": " ".join(str(s) for s in err.seeds),
            })
    csv_path = tables.save_csv(rows, RESDIR, f"degree_table_{device.type}.csv")

    fig, (ax_err, ax_time) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for i, spec in enumerate(map_specs):
        color = f"C{i}"
        by_D = results[spec.name]
        plotting.plot_metric_vs_parameter(
            ax_err, svd_degree.D_VALUES,
            [by_D[D]["error"].mean for D in svd_degree.D_VALUES],
            std=[by_D[D]["error"].std for D in svd_degree.D_VALUES],
            xlabel="D", ylabel="relative Frobenius error", title="accuracy vs. D",
            label=spec.name, color=color,
        )
        plotting.plot_metric_vs_parameter(
            ax_time, svd_degree.D_VALUES,
            [by_D[D]["decomposition_free_s"].mean for D in svd_degree.D_VALUES],
            std=[by_D[D]["decomposition_free_s"].std for D in svd_degree.D_VALUES],
            xlabel="D", ylabel="time (s)", title="decomposition-free time vs. D",
            label=spec.name, color=color,
        )
    ax_err.set_yscale("log")
    fig.suptitle(
        f"Dependence on degree D, {svd_degree.DEFAULT_M}x{svd_degree.DEFAULT_N}, "
        f"{device}/{cli.dtype_name(dtype)}, fixed NS iteration budget"
    )
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig_path = os.path.join(FIGDIR, f"svd_degree_{device.type}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
