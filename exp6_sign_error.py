# exp6_sign_error.py
"""Experiment 6: convergence of the matrix orbit to Sgn(M), d = 1..6."""

import numpy as np
import matplotlib.pyplot as plt

import ns_utils as nu

CFG = {
    "m": 64,
    "n": 48,
    "seed": 0,
    "gen": "gauss",         # 'gauss', or 'edit' to overwrite the spectral tail
    "n_zero": 0,            # 'edit' only: smallest singular values sent to 0
    "n_small": 0,           # 'edit' only: next smallest sent to s_small
    "s_small": 1e-6,
    "normalize": True,
    "d_list": (1, 2, 3, 4, 5, 6),
    "n_iters": 18,
    "norm_method": "svd",   # 'svd' or 'power'
    "norm_power_iters": 300,
    "norm_tol": 1e-12,
    "norm_pad": 1.0 + 1e-6,
    "floor": 1e-18,
    "fit_min": 1e-13,       # below this the pairs are round-off, not convergence
    "fit_max": 5e-1,        # above this the orbit is still in the burn-in phase
    "figsize": (11.0, 4.2),
    "outdir": "figures",
    "fname": "exp6_sign_error.png",
    "dpi": 150,
    "show": True,
}


def order_pairs(cfg, e):
    """Consecutive error pairs (e_k, e_{k+1}) usable for the order plot."""
    x, y = e[:-1], e[1:]
    keep = (x > cfg["fit_min"]) & (y > cfg["fit_min"]) & (x < cfg["fit_max"])
    return x[keep], y[keep]


def run(cfg=CFG):
    M = nu.make_M(cfg)
    sig = nu.svdvals(M)
    N = nu.sgn_svd(M)  # target, read off the SVD; independent of the iteration
    ks = np.arange(cfg["n_iters"] + 1)

    errF, err2 = {}, {}
    for d in cfg["d_list"]:
        Xs = nu.orbit_matrix(M, nu.coeffs_a(d), cfg["n_iters"],
                             norm_method=cfg["norm_method"], norm_kw=nu.norm_kwargs(cfg))
        errF[d] = np.array([float(np.linalg.norm(X - N)) for X in Xs])
        err2[d] = np.array([float(np.linalg.norm(X - N, 2)) for X in Xs])

    pos = sig[sig > nu.rank_tol(M, sig)]
    print("gen=%s, %dx%d, rank %d of %d, ||M||_2=%.6f, sigma_min+=%.3e"
          % (cfg["gen"], cfg["m"], cfg["n"], pos.size, sig.size,
             nu.spec_norm_svd(M), float(np.min(pos))))
    print("%-3s %-13s %-13s" % ("d", "errF final", "err2 final"))
    for d in cfg["d_list"]:
        print("%-3d %-13.3e %-13.3e" % (d, errF[d][-1], err2[d][-1]))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"])

    ax = axes[0]
    for i, d in enumerate(cfg["d_list"]):
        ax.semilogy(ks, np.maximum(errF[d], cfg["floor"]), "o-", ms=3, lw=1.0,
                    color=colors[i % len(colors)], label="d=%d" % d)
    nu.setup_ax(ax, "k", r"$\|X_k-\mathrm{Sgn}(M)\|_F$", "convergence to the sign")

    ax = axes[1]
    for i, d in enumerate(cfg["d_list"]):
        x, y = order_pairs(cfg, err2[d])
        if x.size == 0:
            continue
        c = colors[i % len(colors)]
        ax.loglog(x, y, "o", ms=4, color=c, label="d=%d" % d)
        t = np.array([float(np.min(x)), float(np.max(x))])
        ax.loglog(t, nu.psi_at_one(d) * t ** (d + 1), "--", lw=0.8, color=c)
    nu.setup_ax(ax, r"$\|E_k\|_2$", r"$\|E_{k+1}\|_2$",
                r"order $d+1$; dashed $|\psi_d(1)|\,e^{\,d+1}$")

    nu.output(fig, cfg["outdir"], cfg["fname"], cfg["dpi"], cfg["show"])


if __name__ == "__main__":
    run()