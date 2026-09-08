# exp10_relu_spectrum.py
"""Experiment 10: the matrix hinge (ReLU activation) at a positive threshold."""

import numpy as np
import matplotlib.pyplot as plt

import ns_utils as nu

CFG = {
    "m": 64,
    "n": 48,
    "seed": 0,
    "gen": "edit",          # 'gauss', or 'edit' to overwrite the spectral tail
    "n_zero": 0,
    "n_small": 0,
    "s_small": 1e-6,
    "normalize": True,       # ||M||_2 = 1, so mu scales with sigma
    "mu": 0.35,              # positive threshold, as the experiment asks
    "d": 3,
    "k_show": (1, 2, 3, 5),
    "k_err": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    "xaxis": "sigma",        # 'index' or 'sigma'
    "norm_method": "svd",
    "norm_power_iters": 300,
    "norm_tol": 1e-12,
    "norm_pad": 1.0 + 1e-6,
    "floor": 1e-18,
    "figsize": (11.0, 4.2),
    "outdir": "figures",
    "fname": "exp10_relu_spectrum.png",
    "dpi": 150,
    "show": True,
}


def run(cfg=CFG):
    if cfg["mu"] <= 0.0:
        raise ValueError("this experiment is stated for mu > 0")
    M = nu.make_M(cfg)
    U, sig, V = nu.calc_svd(M)
    f = nu.odd_ext(lambda t: nu.hinge_mu(t, cfg["mu"]))
    Y_ex = nu.op_svd(M, f)
    coords_ex = np.asarray(f(sig))

    k_all = sorted(set(int(k) for k in cfg["k_show"]) | set(int(k) for k in cfg["k_err"]))
    res = nu.sweep_op(lambda k: nu.hinge_map(M, cfg["mu"], nu.sgn_cfg(cfg, k)),
                      k_all, Y_ex, U, V)

    pos = sig[sig > nu.rank_tol(M, sig)]
    print("gen=%s, d=%d, rank %d of %d, sigma_min=%.6f, mu=%.2f"
          % (cfg["gen"], cfg["d"], pos.size, sig.size, float(np.min(pos)), cfg["mu"]))
    print("%-4s %-13s %-13s %-13s" % ("k", "errF", "err2", "resid"))
    for k in cfg["k_err"]:
        r = res[int(k)]
        print("%-4d %-13.3e %-13.3e %-13.3e" % (int(k), r["errF"], r["err2"], r["resid"]))

    fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"])
    nu.plot_spectrum(axes[0], sig, coords_ex,
                     [res[int(k)]["coords"] for k in cfg["k_show"]], cfg["k_show"],
                     r"matrix hinge, $\mu=%.2f$, d=%d" % (cfg["mu"], cfg["d"]),
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