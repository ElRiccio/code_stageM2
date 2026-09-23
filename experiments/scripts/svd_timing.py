"""Reproducible driver for the timing-vs-SVD table/figure (reviewer:
"there should be a direct comparison with the SVD-based implementation:
... computation time").

Run as `python -m experiments.scripts.svd_timing [--device {auto,cpu,cuda}] [--dtype {float32,float64}]`.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments import cli, map_catalogue, svd_timing, tables

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESDIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    device, dtype = cli.parse_device_dtype(
        default_dtype=torch.float64, description="Timing vs. exact SVD, per map."
    )
    map_specs = map_catalogue.all_maps()
    results = svd_timing.timing_experiment(
        map_specs, svd_timing.DEFAULT_M, svd_timing.DEFAULT_N, svd_timing.DEFAULT_D,
        device=device, dtype=dtype,
    )

    rows = []
    for name, quantities in results.items():
        free, svd, speedup = quantities["decomposition_free_s"], quantities["svd_reference_s"], quantities["speedup"]
        rows.append({
            "map": name,
            "device": str(device),
            "dtype": cli.dtype_name(dtype),
            "m": svd_timing.DEFAULT_M,
            "n": svd_timing.DEFAULT_N,
            "D": svd_timing.DEFAULT_D,
            "n_trials": len(free.values),
            "mean_decomposition_free_s": free.mean,
            "std_decomposition_free_s": free.std,
            "mean_svd_reference_s": svd.mean,
            "std_svd_reference_s": svd.std,
            "mean_speedup": speedup.mean,
            "std_speedup": speedup.std,
            "seeds": " ".join(str(s) for s in free.seeds),
        })
    csv_path = tables.save_csv(rows, RESDIR, f"timing_table_{device.type}.csv")

    names = [r["map"] for r in rows]
    free_means = np.array([r["mean_decomposition_free_s"] for r in rows])
    free_stds = np.array([r["std_decomposition_free_s"] for r in rows])
    svd_means = np.array([r["mean_svd_reference_s"] for r in rows])
    svd_stds = np.array([r["std_svd_reference_s"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, free_means * 1e3, width, yerr=free_stds * 1e3, capsize=3, label="decomposition-free (NS)")
    ax.bar(x + width / 2, svd_means * 1e3, width, yerr=svd_stds * 1e3, capsize=3, label="exact SVD reference")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("wall-clock time (ms, min of repeats)")
    ax.set_title(
        f"Timing vs. exact SVD, {svd_timing.DEFAULT_M}x{svd_timing.DEFAULT_N}, "
        f"D={svd_timing.DEFAULT_D}, {device}/{cli.dtype_name(dtype)}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig_path = os.path.join(FIGDIR, f"svd_timing_{device.type}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {fig_path}")
    for r in rows:
        print(
            f"  {r['map']:20s} free={r['mean_decomposition_free_s']*1e3:.3f}ms "
            f"svd={r['mean_svd_reference_s']*1e3:.3f}ms speedup={r['mean_speedup']:.2f}x"
        )


if __name__ == "__main__":
    main()
