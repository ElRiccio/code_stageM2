"""Reproducible driver for the CPU-vs-GPU figure of the SVD-comparison suite
(reviewer: "the report motivates the approach partly by the efficiency of
matrix multiplication on GPUs, but no GPU experiment is presented").

Run as `python -m experiments.scripts.svd_device [--device {both,cpu,cuda}] [--dtype {float32,float64}]`.
`--device both` (the default) runs CPU and CUDA (if available) and produces
the comparison plot; `--device cpu`/`--device cuda` restricts to one device
(a single-device bar chart, no comparison). If CUDA is requested (directly
or via "both") but unavailable, this degrades to a CPU-only table and a
note instead of failing (see experiments/svd_device.py's docstring on why
that is reported, not hidden) -- except `--device cuda` alone, which exits
with a clear error instead of silently running on CPU.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments import cli, map_catalogue, svd_device, tables

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESDIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    requested_devices, dtype = cli.parse_devices_dtype(
        default_dtype=torch.float32, description="CPU vs. GPU accuracy and timing."
    )
    map_specs = map_catalogue.all_maps()
    results = svd_device.device_experiment(map_specs, devices=requested_devices, dtype=dtype)

    # device_experiment silently skips "cuda" when unavailable; the devices
    # that actually ran are read back from the results themselves rather
    # than assumed to match `requested_devices`.
    ran_devices = sorted(next(iter(results.values())).keys())
    tag = "_".join(ran_devices) if ran_devices else "none"

    rows = []
    for map_name, by_device in results.items():
        for device, quantities in by_device.items():
            err, t = quantities["error"], quantities["time_s"]
            rows.append({
                "map": map_name, "device": device, "dtype": cli.dtype_name(dtype),
                "m": svd_device.DEFAULT_SIZE[0], "n": svd_device.DEFAULT_SIZE[1], "D": svd_device.DEFAULT_D,
                "n_trials": len(err.values),
                "mean_relative_frobenius_error": err.mean, "std_relative_frobenius_error": err.std,
                "mean_time_s": t.mean, "std_time_s": t.std,
                "seeds": " ".join(str(s) for s in err.seeds),
            })
    rows.append({
        "map": "", "device": "", "dtype": "", "m": "", "n": "", "D": "", "n_trials": "",
        "mean_relative_frobenius_error": "", "std_relative_frobenius_error": "",
        "mean_time_s": "", "std_time_s": "",
        "seeds": f"cuda_available={torch.cuda.is_available()} requested={requested_devices} ran={ran_devices}",
    })
    csv_path = tables.save_csv(rows, RESDIR, f"device_table_{tag}.csv")
    print(f"wrote {csv_path}")

    if "cuda" in requested_devices and "cuda" not in ran_devices:
        print(
            "No CUDA device available in this environment: table written for "
            f"{ran_devices} only. Re-run on a CUDA-capable machine for the GPU column "
            "(experiments/svd_device.py degrades gracefully by design)."
        )

    names = [spec.name for spec in map_specs]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(len(names))
    if len(ran_devices) >= 2:
        width = 0.8 / len(ran_devices)
        for i, device in enumerate(ran_devices):
            means = np.array([results[n][device]["time_s"].mean for n in names])
            offset = (i - (len(ran_devices) - 1) / 2) * width
            ax.bar(x + offset, means * 1e3, width, label=device.upper())
        ax.legend(fontsize=8)
    elif len(ran_devices) == 1:
        device = ran_devices[0]
        means = np.array([results[n][device]["time_s"].mean for n in names])
        ax.bar(x, means * 1e3, 0.6, color="C0")
    else:
        plt.close(fig)
        return
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("wall-clock time (ms)")
    ax.set_title(
        f"{'CPU vs. GPU' if len(ran_devices) >= 2 else ran_devices[0].upper()}, "
        f"{svd_device.DEFAULT_SIZE[0]}x{svd_device.DEFAULT_SIZE[1]}, D={svd_device.DEFAULT_D}, "
        f"{cli.dtype_name(dtype)}"
    )
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig_path = os.path.join(FIGDIR, f"svd_device_{tag}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
