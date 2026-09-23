"""Reproducible driver for the rank-deficiency / conditioning figure of the
SVD-comparison suite (reviewer: "experiments with several ranks and several
smallest nonzero singular values"; "the influence of conditioning should be
investigated").

Run as `python -m experiments.scripts.svd_rank_sigma`.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

from experiments import map_catalogue, plotting, svd_rank_sigma, tables

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESDIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _rows(results: dict, sweep_name: str) -> list[dict]:
    rows = []
    for map_name, by_value in results.items():
        for value, quantities in by_value.items():
            err, free = quantities["error"], quantities["decomposition_free_s"]
            rows.append({
                "sweep": sweep_name, "map": map_name, "value": value,
                "n_trials": len(err.values),
                "mean_relative_frobenius_error": err.mean, "std_relative_frobenius_error": err.std,
                "mean_decomposition_free_s": free.mean, "std_decomposition_free_s": free.std,
                "seeds": " ".join(str(s) for s in err.seeds),
            })
    return rows


def main() -> None:
    map_specs = map_catalogue.all_maps()
    rank_results = svd_rank_sigma.rank_sweep(map_specs)
    cond_results = svd_rank_sigma.conditioning_sweep(map_specs)

    rows = _rows(rank_results, "rank") + _rows(cond_results, "conditioning")
    csv_path = tables.save_csv(rows, RESDIR, "rank_sigma_table.csv")

    r = min(svd_rank_sigma.DEFAULT_M, svd_rank_sigma.DEFAULT_N)
    ranks = [r - nz for nz in svd_rank_sigma.N_ZERO_VALUES]

    fig, (ax_rank, ax_cond) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for i, spec in enumerate(map_specs):
        color = f"C{i}"
        by_nz = rank_results[spec.name]
        plotting.plot_metric_vs_parameter(
            ax_rank, ranks,
            [by_nz[nz]["error"].mean for nz in svd_rank_sigma.N_ZERO_VALUES],
            std=[by_nz[nz]["error"].std for nz in svd_rank_sigma.N_ZERO_VALUES],
            xlabel=f"rank (out of {r})", ylabel="relative Frobenius error",
            title="accuracy vs. rank", label=spec.name, color=color,
        )
        by_sm = cond_results[spec.name]
        plotting.plot_metric_vs_parameter(
            ax_cond, svd_rank_sigma.SIGMA_MIN_VALUES,
            [by_sm[sm]["error"].mean for sm in svd_rank_sigma.SIGMA_MIN_VALUES],
            std=[by_sm[sm]["error"].std for sm in svd_rank_sigma.SIGMA_MIN_VALUES],
            xlabel=r"$\sigma_{\min}$", ylabel="", title="accuracy vs. $\\sigma_{\\min}$",
            xscale="log", label=spec.name, color=color,
        )
    ax_rank.set_yscale("log")
    ax_cond.set_yscale("log")
    fig.suptitle(
        f"Accuracy vs. SVD under rank deficiency / conditioning, "
        f"{svd_rank_sigma.DEFAULT_M}x{svd_rank_sigma.DEFAULT_N}, D={svd_rank_sigma.DEFAULT_D}"
    )
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig_path = os.path.join(FIGDIR, "svd_rank_sigma.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"wrote {csv_path}")
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
