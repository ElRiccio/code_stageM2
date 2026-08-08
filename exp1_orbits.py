# exp1_orbits.py
"""Experiment 1: iterates of p_d on [-1,1], one figure per degree."""

import numpy as np
import matplotlib.pyplot as plt

import ns_utils as nu

CFG = {
    "d_list": (1, 2, 3, 4, 5, 6),
    "n_grid": 1001,
    "n_iters": 5,
    "x_lim": (-1.0, 1.0),
    "y_lim": (-1.15, 1.15),
    "figsize": (5.0, 4.0),
    "outdir": "figures",
    "dpi": 150,
    "show": True,
}


def run(cfg=CFG):
    x = np.linspace(cfg["x_lim"][0], cfg["x_lim"][1], cfg["n_grid"])
    for d in cfg["d_list"]:
        U = nu.orbit_pd(x, d, cfg["n_iters"])  # rows are the successive iterates
        fig, ax = plt.subplots(figsize=cfg["figsize"])
        ax.plot(x, nu.sgn_exact(x), "k--", lw=1.0, label="sgn")
        for k in range(cfg["n_iters"] + 1):
            ax.plot(x, U[k], lw=1.0, label="k=%d" % k)
        ax.set_xlim(cfg["x_lim"])
        ax.set_ylim(cfg["y_lim"])
        nu.setup_ax(ax, "x", r"$p_d^{\circ k}(x)$", "d=%d, degree %d" % (d, 2 * d + 1))
        nu.output(fig, cfg["outdir"], "exp1_orbit_d%d.png" % d, cfg["dpi"], cfg["show"])


if __name__ == "__main__":
    run()