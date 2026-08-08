# exp7_sign_spectrum.py
"""Experiment 7: the singular values along the sign iteration.

Left panel:  sigma_i(M)/beta dotted, sgn(sigma_i) dashed, and the spectral
             coordinates of X_k at a few k (Cor. scalar-dynamics).
Right panel: distance of X_k to Sgn(M).
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
    "normalize": True,
    "d": 3,
    "k_show": (1, 2, 3, 5),   # iterates drawn in the spectrum panel
    "n_iters": 14,            # length of the error track
    "norm_method": "svd",
    "norm_power_iters": 300,
    "norm_tol": 1e-12,
    "norm_pad": 1.0 + 1e-6,
    "floor": 1e-18,
    "figsize": (11.0, 4.2),
    "outdir": "figures",
    "fname": "exp7_sign_spectrum.png",
    "dpi": 150,
    "show": True,
}


def run(cfg=CFG):
    M = nu.make_M(cfg)
    U, sig, V = nu.calc_svd(M)
    beta = nu.pre_norm(M, cfg)
    tgt = (sig > nu.rank_tol(M, sig)).astype(float)  # sgn(sigma), with sgn(0)=0
    N = nu.sgn_svd(M)

    n_max = max(int(max(cfg["k_show"])), int(cfg["n_iters"]))
    Xs = nu.orbit_matrix(M, nu.coeffs_a(cfg["d"]), n_max, scale=beta)
    coords = [nu.spec_coords(Xs[int(k)], U, V) for k in cfg["k_show"]]

    ks = np.arange(cfg["n_iters"] + 1)
    errF = np.array([float(np.linalg.norm(Xs[int(k)] - N)) for k in ks])
    err2 = np.array([float(np.linalg.norm(Xs[int(k)] - N, 2)) for k in ks])

    print("gen=%s, d=%d, rank %d of %d, beta=%.6f"
          % (cfg["gen"], cfg["d"], int(np.sum(tgt)), sig.size, beta))
    print("%-4s %-14s %-16s" % ("k", "frame resid", "|sorted-svdvals|"))
    for k in cfg["k_show"]:
        X = Xs[int(k)]
        c = nu.spec_coords(X, U, V)
        # p_d is increasing on [0,1], so the ordering is preserved and the
        # spectral coordinates are exactly the singular values of X_k.
        dev = float(np.max(np.abs(np.sort(c)[::-1] - nu.svdvals(X))))
        print("%-4d %-14.3e %-16.3e" % (int(k), nu.frame_resid(X, U, V), dev))

    fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"])
    nu.plot_spectrum(axes[0], sig / beta, tgt, coords, cfg["k_show"],
                     "sign iteration, d=%d" % cfg["d"],
                     ylabel=r"singular value of $X_k$")
    nu.plot_decay(axes[1], ks, [errF, err2], [r"$\|\cdot\|_F$", r"$\|\cdot\|_2$"],
                  r"distance to $\mathrm{Sgn}(M)$", floor=cfg["floor"])
    nu.output(fig, cfg["outdir"], cfg["fname"], cfg["dpi"], cfg["show"])


if __name__ == "__main__":
    run()
