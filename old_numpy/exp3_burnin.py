# exp3_burnin.py
"""Experiment 3: one degree, several u_0, showing the burn-in phase."""

import numpy as np
import matplotlib.pyplot as plt

import ns_utils as nu

CFG = {
    "d": 1,                # 1 (cubic) or 2 (quintic)
    "u0_list": (1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 0.5, 0.9),
    "n_iters": 45,
    "l_floor": -320.0,
    "n_grid_const": 20001,
    "figsize": (6.5, 4.5),
    "outdir": "figures",
    "dpi": 150,
    "show": True,
}


def run(cfg=CFG):
    d = cfg["d"]
    mu, rho, lam = nu.constants(d, cfg["n_grid_const"])
    print("d=%d  mu_d=%.6f  rho_d=%.6f  lambda_d=%.6f" % (d, mu, rho, lam))
    print("%10s %12s %10s" % ("u0", "k* measured", "k* bound"))

    fig, ax = plt.subplots(figsize=cfg["figsize"])
    for u0 in cfg["u0_list"]:
        L = nu.log10_err_orbit(u0, d, cfg["n_iters"])
        ks = nu.entry_index(L, d, cfg["n_grid_const"])
        print("%10.1e %12d %10d" % (u0, ks, nu.burnin_bound(u0, d, cfg["n_grid_const"])))

        Lp = np.maximum(L, cfg["l_floor"])  # clamped for display only
        line = ax.plot(np.arange(L.size), Lp, "-", lw=1.0, label="u0=%.1e" % u0)[0]
        if ks >= 0:
            ax.plot([ks], [Lp[ks]], "o", ms=5, color=line.get_color())

    # ax.axhline(np.log10(rho), color="k", ls="--", lw=1.0, label=r"$\log_{10}\rho_d$ (entry)")
    ax.set_ylim(cfg["l_floor"], 0.5)
    nu.setup_ax(ax, "k", r"$\log_{10}|1-u_k|$",
                "burn-in then order %d (d=%d); markers = entry index" % (d + 1, d))
    nu.output(fig, cfg["outdir"], "exp3_burnin_d%d.png" % d, cfg["dpi"], cfg["show"])


if __name__ == "__main__":
    run()