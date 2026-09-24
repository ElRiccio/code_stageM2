# exp11_cpwl_spectrum.py
"""Experiment 11: a continuous piecewise-linear profile through Sgn alone."""

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
    "normalize": True,       # ||M||_2 = 1, so the knots scale with sigma
    "knots": (0.0, 0.20, 0.45, 0.70, 1.00),   # 5 knots, hence 3 breakpoints
    "vals": (0.0, 0.10, 0.50, 0.40, 0.90),    # first value 0, so eta = 0
    "d": 1,
    "k_show": (1, 2, 3, 4, 5),
    "k_err": (1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50),
    "xaxis": "sigma",        # 'index' or 'sigma'
    "norm_method": "svd",
    "norm_power_iters": 300,
    "norm_tol": 1e-12,
    "norm_pad": 1.0 + 1e-6,
    "floor": 1e-18,
    "figsize": (11.0, 4.2),
    "outdir": "figures",
    "fname": "exp11_cpwl_spectrum.png",
    "dpi": 150,
    "show": True,
}


def run(cfg=CFG):
    M = nu.make_M(cfg)
    spl = nu.Spline(cfg["knots"], cfg["vals"])
    if abs(spl.eta) > 1e-12:
        raise ValueError("place the first knot at 0 with value 0, so that eta = 0")
    U, sig, V = nu.calc_svd(M)
    Y_ex = nu.op_svd(M, spl.odd_ext)     # reference, read off the SVD
    coords_ex = np.asarray(spl.odd_ext(sig))

    k_all = sorted(set(int(k) for k in cfg["k_show"]) | set(int(k) for k in cfg["k_err"]))
    res = nu.sweep_op(lambda k: nu.spline_map(M, spl, nu.sgn_cfg(cfg, k)),
                      k_all, Y_ex, U, V)

    pos = sig[sig > nu.rank_tol(M, sig)]
    print("gen=%s, d=%d, rank %d of %d, sigma_min=%.6f, knots=%d"
          % (cfg["gen"], cfg["d"], pos.size, sig.size, float(np.min(pos)),
             spl.interior.size))
    print("%-4s %-13s %-13s %-13s" % ("k", "errF", "err2", "resid"))
    for k in cfg["k_err"]:
        r = res[int(k)]
        print("%-4d %-13.3e %-13.3e %-13.3e" % (int(k), r["errF"], r["err2"], r["resid"]))

    fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"])
    nu.plot_spectrum(axes[0], sig, coords_ex,
                     [res[int(k)]["coords"] for k in cfg["k_show"]], cfg["k_show"],
                     "CPWL spline, %d breakpoints, d=%d" % (spl.interior.size, cfg["d"]),
                     xaxis=cfg["xaxis"])
    for xi in spl.interior:
        axes[0].axhline(float(spl.ext(xi)), color="0.8", lw=0.5)
    ks = [int(k) for k in k_all]
    nu.plot_decay(axes[1], ks,
                  [[res[k]["errF"] for k in ks], [res[k]["err2"] for k in ks]],
                  [r"$\|\cdot\|_F$", r"$\|\cdot\|_2$"],
                  "distance to the exact operator", floor=cfg["floor"])
    nu.output(fig, cfg["outdir"], cfg["fname"], cfg["dpi"], cfg["show"])


if __name__ == "__main__":
    run()
