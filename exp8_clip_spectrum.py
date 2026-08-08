# exp8_clip_spectrum.py
"""Experiment 8: spectral clipping through Sgn alone (Thm. clip).

The argument may be rank deficient, so Prop. clip-well-posed leaves only the
readings that hold on all of R^{m x n}: alpha <= 0 <= beta, which is what is
configured. Singular values being nonnegative, the lower branch is then inert on
the spectrum and the operator caps at beta; the alpha branch lives on the odd
extension, i.e. on negative diagonal entries of a signed SVD.

Left panel:  sigma_i(M) dotted, clip_odd(sigma_i) dashed, NS values at a few k.
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
    "normalize": True,       # ||M||_2 = 1, so alpha and beta scale with sigma
    "alpha": 0.30,          # require alpha <= 0 <= beta
    "beta": 0.60,
    "d": 3,
    "k_show": (1, 2, 3, 5),
    "k_err": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    "norm_method": "svd",
    "norm_power_iters": 300,
    "norm_tol": 1e-12,
    "norm_pad": 1.0 + 1e-6,
    "floor": 1e-18,
    "figsize": (11.0, 4.2),
    "outdir": "figures",
    "fname": "exp8_clip_spectrum.png",
    "dpi": 150,
    "show": True,
}


def run(cfg=CFG):
    # if not cfg["alpha"] <= 0.0 <= cfg["beta"]:
    #     raise ValueError("choose alpha <= 0 <= beta: the argument may be rank deficient")
    M = nu.make_M(cfg)
    U, sig, V = nu.calc_svd(M)
    f = nu.odd_ext(lambda t: nu.clip_ab(t, cfg["alpha"], cfg["beta"]))
    Y_ex = nu.op_svd(M, f)          # reference, read off the SVD
    coords_ex = np.asarray(f(sig))

    k_all = sorted(set(int(k) for k in cfg["k_show"]) | set(int(k) for k in cfg["k_err"]))
    res = nu.sweep_op(
        lambda k: nu.clip_map(M, cfg["alpha"], cfg["beta"], nu.sgn_cfg(cfg, k)),
        k_all, Y_ex, U, V)

    pos = sig[sig > nu.rank_tol(M, sig)]
    print("gen=%s, clip [%.2f,%.2f], d=%d, rank %d of %d, gap to beta %.3e"
          % (cfg["gen"], cfg["alpha"], cfg["beta"], cfg["d"], pos.size, sig.size,
             float(np.min(np.abs(pos - cfg["beta"])))))
    print("%-4s %-13s %-13s %-13s" % ("k", "errF", "err2", "frame resid"))
    for k in cfg["k_err"]:
        r = res[int(k)]
        print("%-4d %-13.3e %-13.3e %-13.3e" % (int(k), r["errF"], r["err2"], r["resid"]))

    fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"])
    nu.plot_spectrum(axes[0], sig, coords_ex,
                     [res[int(k)]["coords"] for k in cfg["k_show"]], cfg["k_show"],
                     r"clipping, $[\alpha,\beta]=[%.2f,%.2f]$, d=%d"
                     % (cfg["alpha"], cfg["beta"], cfg["d"]))
    axes[0].axhline(cfg["beta"], color="0.7", lw=0.6)
    ks = [int(k) for k in k_all]
    nu.plot_decay(axes[1], ks,
                  [[res[k]["errF"] for k in ks], [res[k]["err2"] for k in ks]],
                  [r"$\|\cdot\|_F$", r"$\|\cdot\|_2$"],
                  "distance to the exact operator", floor=cfg["floor"])
    nu.output(fig, cfg["outdir"], cfg["fname"], cfg["dpi"], cfg["show"])


if __name__ == "__main__":
    run()
