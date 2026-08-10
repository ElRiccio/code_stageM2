# exp8_clip_spectrum.py
"""Experiment 8: spectral clipping through Sgn alone (Thm. clip)."""

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
    "normalize": True,       # ||M||_2 = 1, so alpha and beta scale with sigma
    "alpha": -0.30,
    "beta": 0.60,
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
    "fname": "exp8_clip_spectrum.png",
    "dpi": 150,
    "show": True,
}


def run(cfg=CFG):
    M = nu.make_M(cfg)
    U, sig, V = nu.calc_svd(M)
    pos = sig[sig > nu.rank_tol(M, sig)]
    if pos.size < sig.size and not cfg["alpha"] <= 0.0 <= cfg["beta"]:
        raise ValueError("rank-deficient argument: choose alpha <= 0 <= beta")

    f = nu.odd_ext(lambda t: nu.clip_ab(t, cfg["alpha"], cfg["beta"]))
    Y_ex = nu.op_svd(M, f)          # reference, read off the SVD
    coords_ex = np.asarray(f(sig))

    k_all = sorted(set(int(k) for k in cfg["k_show"]) | set(int(k) for k in cfg["k_err"]))
    res = nu.sweep_op(
        lambda k: nu.clip_map(M, cfg["alpha"], cfg["beta"], nu.sgn_cfg(cfg, k)),
        k_all, Y_ex, U, V)

    print("gen=%s, d=%d, rank %d of %d, sigma_min=%.6f, alpha=%.2f, beta=%.2f"
          % (cfg["gen"], cfg["d"], pos.size, sig.size, float(np.min(pos)),
             cfg["alpha"], cfg["beta"]))
    print("%-4s %-13s %-13s %-13s" % ("k", "errF", "err2", "resid"))
    for k in cfg["k_err"]:
        r = res[int(k)]
        print("%-4d %-13.3e %-13.3e %-13.3e" % (int(k), r["errF"], r["err2"], r["resid"]))

    fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"])
    nu.plot_spectrum(axes[0], sig, coords_ex,
                     [res[int(k)]["coords"] for k in cfg["k_show"]], cfg["k_show"],
                     r"clipping, $[\alpha,\beta]=[%.2f,%.2f]$, d=%d"
                     % (cfg["alpha"], cfg["beta"], cfg["d"]), xaxis=cfg["xaxis"])
    axes[0].axhline(cfg["beta"], color="0.7", lw=0.6)
    ks = [int(k) for k in k_all]
    nu.plot_decay(axes[1], ks,
                  [[res[k]["errF"] for k in ks], [res[k]["err2"] for k in ks]],
                  [r"$\|\cdot\|_F$", r"$\|\cdot\|_2$"],
                  "distance to the exact operator", floor=cfg["floor"])
    nu.output(fig, cfg["outdir"], cfg["fname"], cfg["dpi"], cfg["show"])


if __name__ == "__main__":
    run()
