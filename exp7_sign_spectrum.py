# exp7_sign_spectrum.py
"""Experiment 7: the singular values along the sign iteration."""

import numpy as np
import matplotlib.pyplot as plt

import ns_utils as nu

CFG = {
    "m": 64,
    "n": 48,
    "seed": 0,
    "gen": "edit",           # 'gauss', or 'edit' to overwrite the spectral tail
    "n_zero": 1,
    "n_small": 0,
    "s_small": 1e-10,
    "normalize": True,
    "d": 3,
    "k_show": (1, 2, 3, 5),   # iterates drawn in the spectrum panel
    "n_iters": 14,            # length of the error track
    "xaxis": "sigma",         # 'index' or 'sigma'
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
    "tol": 1e-12
}


def run(cfg=CFG):
    M = nu.make_M(cfg)
    U, sig, V = nu.calc_svd(M)
    # print("sig:", sig)
    beta = nu.pre_norm(M, cfg)
    tgt = (sig > nu.rank_tol(M, sig)).astype(float)  # sgn(sigma), with sgn(0)=0
    # print("rank_tol:", nu.rank_tol(M, sig))
    # print("tgt:", tgt)
    N = nu.sgn_svd(M)

    n_max = max(int(max(cfg["k_show"])), int(cfg["n_iters"]))
    Xs = nu.orbit_matrix(M, nu.coeffs_a(cfg["d"]), n_max, scale=beta, tol=cfg.get("tol"))
    coords = [nu.spec_coords(Xs[int(k)], U, V) for k in cfg["k_show"]]
    # print("coords[-1]:", coords[-1])

    ks = np.arange(cfg["n_iters"] + 1)
    errF = np.array([float(np.linalg.norm(Xs[int(k)] - N)) for k in ks])
    err2 = np.array([float(np.linalg.norm(Xs[int(k)] - N, 2)) for k in ks])
    resid = np.array([nu.frame_resid(Xs[int(k)], U, V) for k in ks])

    pos = sig[sig > nu.rank_tol(M, sig)]
    print("gen=%s, d=%d, rank %d of %d, sigma_min=%.6f"
          % (cfg["gen"], cfg["d"], pos.size, sig.size, float(np.min(pos))))
    print("%-4s %-13s %-13s %-13s" % ("k", "errF", "err2", "resid"))
    for k in ks:
        print("%-4d %-13.3e %-13.3e %-13.3e" % (int(k), errF[k], err2[k], resid[k]))

    fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"])
    nu.plot_spectrum(axes[0], sig / beta, tgt, coords, cfg["k_show"],
                     "sign iteration, d=%d" % cfg["d"], xaxis=cfg["xaxis"],
                     ylabel=r"singular value of $X_k$")
    nu.plot_decay(axes[1], ks, [errF, err2], [r"$\|\cdot\|_F$", r"$\|\cdot\|_2$"],
                  r"distance to $\mathrm{Sgn}(M)$", floor=cfg["floor"])
    nu.output(fig, cfg["outdir"], cfg["fname"], cfg["dpi"], cfg["show"])


if __name__ == "__main__":
    run()
