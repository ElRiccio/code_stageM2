# exp5_cpwl.py
"""Experiment 5: a CPWL map with two interior knots, from the iterated sign."""

import numpy as np
import matplotlib.pyplot as plt

import ns_utils as nu

CFG = {
    "d": 2,
    "ns_list": (1, 2, 3, 5, 8),
    "knots": (0.0, 0.5, 1.2, 2.0),   # interior knots (breakpoints): 0.5 and 1.2
    "vals": (0.0, 0.8, 0.4, 1.4),
    "x_lim": (-2.5, 2.5),
    "n_grid": 1601,
    "figsize": (10.0, 4.2),
    "outdir": "figures",
    "dpi": 150,
    "show": True,
}


def run(cfg=CFG):
    s = nu.Spline(cfg["knots"], cfg["vals"])
    print("cell slopes c =", np.round(s.c, 6))
    print("slope jumps w =", np.round(s.w, 6))
    print("eta=%.6f  theta=%.6f  kappa=%.6f" % (s.eta, s.theta, s.kappa))

    x = np.linspace(cfg["x_lim"][0], cfg["x_lim"][1], cfg["n_grid"])
    ext_ref = s.ext(x)      # ReLU representation, the reference
    odd_ref = s.odd_ext(x)
    # The sign and ReLU representations must agree at the exact sign.
    print("sign-form vs ReLU-form residual: %.3e"
          % np.max(np.abs(s.ext_sgn(x, nu.sgn_exact) - ext_ref)))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=cfg["figsize"])
    print("%4s %14s %16s" % ("ns", "sup err ext", "sup err odd-ext"))
    for ns in cfg["ns_list"]:
        sgn = nu.make_sgn(cfg["d"], ns)
        e = s.ext_sgn(x, sgn)
        o = s.odd_ext_sgn(x, sgn)
        print("%4d %14.3e %16.3e" % (ns, np.max(np.abs(e - ext_ref)),
                                     np.max(np.abs(o - odd_ref))))
        ax0.plot(x, e, lw=1.0, label="ns=%d" % ns)
        ax1.plot(x, o, lw=1.0, label="ns=%d" % ns)

    ax0.plot(x, ext_ref, "k--", lw=1.0, label="exact")
    ax1.plot(x, odd_ref, "k--", lw=1.0, label="exact")
    for xi in s.interior:
        ax0.axvline(xi, color="grey", lw=0.5)
        ax1.axvline(xi, color="grey", lw=0.5)
        ax1.axvline(-xi, color="grey", lw=0.5)
    nu.setup_ax(ax0, "x", r"$\hat{s}(x)$", "canonical extension (d=%d)" % cfg["d"])
    nu.setup_ax(ax1, "x", r"$\mathrm{sgn}(x)\,\hat{s}(|x|)$",
                "odd extension, outer sign also iterated (d=%d)" % cfg["d"])
    nu.output(fig, cfg["outdir"], "exp5_cpwl.png", cfg["dpi"], cfg["show"])


if __name__ == "__main__":
    run()