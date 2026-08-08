# exp11_cpwl_spectrum.py
"""Experiment 11: a continuous piecewise-linear profile through Sgn alone.

Validates Thm. spline and Prop. spline-sign-form. Only the values on the
half-line are ever used, so the first knot sits at the origin with s(0)=0 as in
eq. net-normalization; then eta = 0, the expression reduces to
c_0 M + sum_i w_i H_{x_i}(M), and no hypothesis beyond that is needed on a
rank-deficient argument. The knot values are nondecreasing, so the spectral
coordinates plotted below are also the singular values of the output.

Left panel:  sigma_i(M) dotted, s_odd(sigma_i) dashed, NS values at a few k.
Right panel: distance of the NS expression to the exact operator, against k.
"""

import numpy as np
import matplotlib.pyplot as plt

import ns_utils as nu

CFG = {
    "m": 64,
    "n": 48,
    "seed": 0,
    "gen": "gauss",
    "n_zero": 8,
    "s_min": 1e-2,
    "s_max": 1.0,
    "normalize": True,       # ||M||_2 = 1, so the knots scale with sigma
    "knots": (0.0, 0.20, 0.45, 0.70, 1.00),   # 5 knots, hence 3 breakpoints
    "vals": (0.0, 0.10, 0.50, 0.80, 0.90),    # first value 0, so eta = 0
    "d": 3,
    "k_show": (1, 2, 3, 5),
    "k_err": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    "k_check": 12,
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

    Sgn = nu.sgn_cfg(cfg, cfg["k_check"])
    # Both forms rearrange one and the same set of sign evaluations, so they must
    # agree to round-off even when Sgn is only the surrogate.
    gap = float(np.linalg.norm(nu.spline_map(M, spl, Sgn)
                               - nu.spline_map_sign(M, spl, Sgn)))

    print("gen=%s, %d interior knots at %s"
          % (cfg["gen"], spl.interior.size, np.array2string(spl.interior, precision=3)))
    print("c = %s, w = %s" % (np.array2string(spl.c, precision=4),
                              np.array2string(spl.w, precision=4)))
    print("eta=%.3e, theta=%.4f, kappa=%.4f, d=%d, rank %d of %d"
          % (spl.eta, spl.theta, spl.kappa, cfg["d"],
             int(np.sum(sig > nu.rank_tol(M, sig))), sig.size))
    print("hinge form vs sign form (Prop. spline-sign-form), k=%d: %.3e"
          % (cfg["k_check"], gap))
    print("%-4s %-13s %-13s %-13s %-16s"
          % ("k", "errF", "err2", "frame resid", "|sorted-svdvals|"))
    for k in cfg["k_err"]:
        r = res[int(k)]
        dev = float(np.max(np.abs(np.sort(r["coords"])[::-1] - nu.svdvals(r["Y"]))))
        print("%-4d %-13.3e %-13.3e %-13.3e %-16.3e"
              % (int(k), r["errF"], r["err2"], r["resid"], dev))

    fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"])
    nu.plot_spectrum(axes[0], sig, coords_ex,
                     [res[int(k)]["coords"] for k in cfg["k_show"]], cfg["k_show"],
                     "CPWL spline, %d breakpoints, d=%d" % (spl.interior.size, cfg["d"]))
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
