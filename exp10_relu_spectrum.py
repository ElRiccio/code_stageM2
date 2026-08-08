# exp10_relu_spectrum.py
"""Experiment 10: the matrix hinge (ReLU activation) at a positive threshold.

Validates Lem. mhinge: H_mu acts on the singular values by the odd hinge, which
for mu >= 0 is ReLU(sigma - mu). Two further identities are checked numerically:
Lem. odd-hinge (ii), the odd hinge at mu > 0 is soft-thresholding, and
Prop. mhinge-neg, at mu <= 0 the inner sign collapses to H_mu(M) = M - mu Sgn(M).

Left panel:  sigma_i(M) dotted, ReLU(sigma_i - mu) dashed, NS values at a few k.
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
    "normalize": True,       # ||M||_2 = 1, so mu scales with sigma
    "mu": 0.35,              # positive threshold, as the experiment asks
    "mu_neg": -0.20,         # nonpositive threshold, for the collapse check
    "d": 3,
    "k_show": (1, 2, 3, 5),
    "k_err": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    "k_check": 12,           # NS budget used for the two identity checks
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

    Sgn = nu.sgn_cfg(cfg, cfg["k_check"])
    gap_soft = float(np.linalg.norm(
        nu.hinge_map(M, cfg["mu"], Sgn) - nu.op_svd(M, lambda t: nu.soft_g(t, cfg["mu"]))))
    gap_neg = float(np.linalg.norm(
        nu.hinge_map(M, cfg["mu_neg"], Sgn) - nu.hinge_map_neg(M, cfg["mu_neg"], Sgn)))

    pos = sig[sig > nu.rank_tol(M, sig)]
    print("gen=%s, mu=%.3f, d=%d, rank %d of %d, gap to mu %.3e"
          % (cfg["gen"], cfg["mu"], cfg["d"], pos.size, sig.size,
             float(np.min(np.abs(pos - cfg["mu"])))))
    print("H_mu vs exact soft_mu (Lem. odd-hinge (ii)), k=%d: %.3e"
          % (cfg["k_check"], gap_soft))
    print("H_mu vs M - mu Sgn(M) at mu=%.2f (Prop. mhinge-neg): %.3e"
          % (cfg["mu_neg"], gap_neg))
    print("%-4s %-13s %-13s %-13s" % ("k", "errF", "err2", "frame resid"))
    for k in cfg["k_err"]:
        r = res[int(k)]
        print("%-4d %-13.3e %-13.3e %-13.3e" % (int(k), r["errF"], r["err2"], r["resid"]))

    fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"])
    nu.plot_spectrum(axes[0], sig, coords_ex,
                     [res[int(k)]["coords"] for k in cfg["k_show"]], cfg["k_show"],
                     r"matrix hinge, $\mu=%.2f$, d=%d" % (cfg["mu"], cfg["d"]))
    axes[0].axhline(0.0, color="0.7", lw=0.6)
    ks = [int(k) for k in k_all]
    nu.plot_decay(axes[1], ks,
                  [[res[k]["errF"] for k in ks], [res[k]["err2"] for k in ks]],
                  [r"$\|\cdot\|_F$", r"$\|\cdot\|_2$"],
                  "distance to the exact operator", floor=cfg["floor"])
    nu.output(fig, cfg["outdir"], cfg["fname"], cfg["dpi"], cfg["show"])


if __name__ == "__main__":
    run()
