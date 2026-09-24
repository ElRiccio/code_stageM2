# exp9_soft_spectrum.py
"""Experiment 9: the nuclear-norm proximal operator through Sgn alone."""

import numpy as np
import matplotlib.pyplot as plt

import ns_utils as nu

CFG = {
    "m": 64,
    "n": 48,
    "seed": 0,
    "gen": "edit",          # 'gauss', or 'edit' to overwrite the spectral tail
    "n_zero": 0,
    "n_small": 1,
    "s_small": 1e-6,
    "normalize": True,       # ||M||_2 = 1, so gamma scales with sigma
    "gamma": 0.25,
    "d": 3,
    "k_show": (11, 12, 13, 14, 15),
    "k_err": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
    "xaxis": "sigma",        # 'index' or 'sigma'
    "norm_method": "svd",
    "norm_power_iters": 300,
    "norm_tol": 1e-12,
    "norm_pad": 1.0 + 1e-6,
    "floor": 1e-18,
    "figsize": (11.0, 4.2),
    "outdir": "figures",
    "fname": "exp9_soft_spectrum.png",
    "dpi": 150,
    "show": True,
    "tol": 1e-12
}


def run(cfg=CFG):
    M = nu.make_M(cfg)
    U, sig, V = nu.calc_svd(M)
    f = lambda t: nu.soft_g(t, cfg["gamma"])  # odd already, hence its own odd extension
    Y_ex = nu.op_svd(M, f)
    coords_ex = np.asarray(f(sig))

    k_all = sorted(set(int(k) for k in cfg["k_show"]) | set(int(k) for k in cfg["k_err"]))
    res = nu.sweep_op(lambda k: nu.soft_map(M, cfg["gamma"], nu.sgn_cfg(cfg, k)),
                      k_all, Y_ex, U, V)

    pos = sig[sig > nu.rank_tol(M, sig)]
    print("gen=%s, d=%d, rank %d of %d, sigma_min=%.6f, gamma=%.3f"
          % (cfg["gen"], cfg["d"], pos.size, sig.size, float(np.min(pos)), cfg["gamma"]))
    print("%-4s %-13s %-13s %-13s" % ("k", "errF", "err2", "resid"))
    for k in cfg["k_err"]:
        r = res[int(k)]
        print("%-4d %-13.3e %-13.3e %-13.3e" % (int(k), r["errF"], r["err2"], r["resid"]))

    fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"])
    nu.plot_spectrum(axes[0], sig, coords_ex,
                     [res[int(k)]["coords"] for k in cfg["k_show"]], cfg["k_show"],
                     r"nuclear prox, $\gamma=%.2f$, d=%d" % (cfg["gamma"], cfg["d"]),
                     xaxis=cfg["xaxis"])
    axes[0].axhline(0.0, color="0.7", lw=0.6)
    ks = [int(k) for k in k_all]
    nu.plot_decay(axes[1], ks,
                  [[res[k]["errF"] for k in ks], [res[k]["err2"] for k in ks]],
                  [r"$\|\cdot\|_F$", r"$\|\cdot\|_2$"],
                  "distance to the exact operator", floor=cfg["floor"])
    nu.output(fig, cfg["outdir"], cfg["fname"], cfg["dpi"], cfg["show"])


if __name__ == "__main__":
    run()