# exp2_order.py
"""Experiment 2: order d+1 from an initial point already inside the ball."""

import numpy as np
import matplotlib.pyplot as plt

import ns_utils as nu

CFG = {
    "d_list": (1, 2, 3, 4, 5, 6),
    "u0": 0.9,             # inside {|x-1| < rho_d} for every d listed
    "n_iters": 12,
    "l_floor": -400.0,     # keeps the axes on a readable range
    "n_grid_const": 20001,
    "figsize": (10.0, 4.2),
    "outdir": "figures",
    "dpi": 150,
    "show": True,
}


def run(cfg=CFG):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=cfg["figsize"])
    for d in cfg["d_list"]:
        rho = nu.constants(d, cfg["n_grid_const"])[1]
        if abs(cfg["u0"] - 1.0) >= rho:
            print("warning: d=%d, |u0-1|=%.3f >= rho_d=%.3f" % (d, abs(cfg["u0"] - 1.0), rho))
        L = nu.log10_err_orbit(cfg["u0"], d, cfg["n_iters"])
        A = np.log10(nu.psi_at_one(d))

        # (a) successive errors
        keep = (L[:-1] >= cfg["l_floor"]) & (L[1:] >= cfg["l_floor"])
        xs = L[:-1][keep]
        ys = L[1:][keep]
        line = ax0.plot(xs, ys, "o-", ms=3, lw=1.0, label="d=%d" % d)[0]
        if xs.size > 0:
            xr = np.array([xs.min(), xs.max()])
            ax0.plot(xr, (d + 1) * xr + A, ":", lw=1.0, color=line.get_color())

        # (b) error against iteration count
        ax1.plot(np.arange(L.size), np.maximum(L, cfg["l_floor"]), "o-", ms=3,
                 lw=1.0, label="d=%d" % d)


    nu.setup_ax(ax0, r"$\log_{10}|1-u_k|$", r"$\log_{10}|1-u_{k+1}|$",
                "successive errors, slope $d+1$ (u0=%.2f)" % cfg["u0"])
    ax1.set_ylim(cfg["l_floor"], 0.5)
    nu.setup_ax(ax1, "k", r"$\log_{10}|1-u_k|$", "error decay (u0=%.2f)" % cfg["u0"])
    nu.output(fig, cfg["outdir"], "exp2_order.png", cfg["dpi"], cfg["show"])


if __name__ == "__main__":
    run()