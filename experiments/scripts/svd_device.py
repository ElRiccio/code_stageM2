"""Reproducible driver for the CPU-vs-GPU figure of the SVD-comparison suite
(reviewer: "the report motivates the approach partly by the efficiency of
matrix multiplication on GPUs, but no GPU experiment is presented").

Run as `python -m experiments.scripts.svd_device`. If no CUDA device is
available, writes the CPU-only table and a note instead of a comparison
plot (see experiments/svd_device.py's docstring on why that is reported,
not hidden).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments import map_catalogue, svd_device, tables

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESDIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    cuda_available = torch.cuda.is_available()
    map_specs = map_catalogue.all_maps()
    results = svd_device.device_experiment(map_specs)

    rows = []
    for map_name, by_device in results.items():
        for device, quantities in by_device.items():
            err, t = quantities["error"], quantities["time_s"]
            rows.append({
                "map": map_name, "device": device,
                "m": svd_device.DEFAULT_SIZE[0], "n": svd_device.DEFAULT_SIZE[1], "D": svd_device.DEFAULT_D,
                "n_trials": len(err.values),
                "mean_relative_frobenius_error": err.mean, "std_relative_frobenius_error": err.std,
                "mean_time_s": t.mean, "std_time_s": t.std,
                "seeds": " ".join(str(s) for s in err.seeds),
            })
    rows.append({
        "map": "", "device": "", "m": "", "n": "", "D": "", "n_trials": "",
        "mean_relative_frobenius_error": "", "std_relative_frobenius_error": "",
        "mean_time_s": "", "std_time_s": "",
        "seeds": f"cuda_available={cuda_available}",
    })
    csv_path = tables.save_csv(rows, RESDIR, "device_table.csv")
    print(f"wrote {csv_path}")

    if not cuda_available:
        print(
            "No CUDA device available in this environment: CPU-only table written; "
            "no CPU-vs-GPU comparison plot produced. Re-run on a CUDA-capable machine "
            "for the GPU column (experiments/svd_device.py degrades gracefully by design)."
        )
        return

    names = [spec.name for spec in map_specs]
    cpu_means = np.array([results[n]["cpu"]["time_s"].mean for n in names])
    cuda_means = np.array([results[n]["cuda"]["time_s"].mean for n in names])

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, cpu_means * 1e3, width, label="CPU")
    ax.bar(x + width / 2, cuda_means * 1e3, width, label="CUDA")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("wall-clock time (ms)")
    ax.set_title(f"CPU vs. GPU, {svd_device.DEFAULT_SIZE[0]}x{svd_device.DEFAULT_SIZE[1]}, D={svd_device.DEFAULT_D}")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig_path = os.path.join(FIGDIR, "svd_device.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
