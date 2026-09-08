# exp16_sign_spectrum.py
"""Experiment 16: the sign iteration read on the three cases.

Rows: a symmetric argument, read on its eigenvalues; a rectangular argument of
full rank, read on its singular values; the same with one singular value set to
zero. Columns: the spectral coordinates of X_k against those of the argument,
and the distance to msgn(M).

The third row is the point of the figure. A vanishing singular value is an exact
fixed point of every odd polynomial at the prescribed value sgn(0) = 0, so the
iteration converges there with no rank hypothesis, and msgn is not the polar
factor.
"""

import numpy as np
import matplotlib.pyplot as plt

import ns_utils as nu

CFG = {
    "m": 64,
    "n": 48,
    "seed": 0,
    "gen": "edit",
    "n_small": 0,
    "s_small": 1e-6,
    "normalize": True,
    "d": 3,              # displayed as D; the key stays "d" for ns_utils
    "k_show": (1, 2, 3, 5),
    "n_iters": 16,
    "norm_method": "svd",
    "norm_power_iters": 300,
    "norm_tol": 1e-12,
    "norm_pad": 1.0 + 1e-6,
    "floor": 1e-18,
    "figsize": (10.0, 10.2),
    "outdir": "figures",
    "fname": "exp16_sign_spectrum.png",
    "dpi": 150,
    "show": True,
    "tol": 1e-12,
}


def plot_spec(ax, xs, exact, approx, ks, title, xlab, ylab):
    """Input spectrum (dotted), target (dashed) and the values at a few k."""
    xs = np.asarray(xs, dtype=float)
    order = np.argsort(xs)
    x = xs[order]
    ax.plot(x, x, ":", color="0.35", lw=1.2, label="input")
    for k, v in zip(ks, approx):
        ax.plot(x, np.asarray(v, dtype=float)[order], lw=1.0, label="k=%d" % int(k))
    ax.plot(x, np.asarray(exact, dtype=float)[order], "k--", lw=1.3,
            label=r"$\mathrm{msgn}(M)$")
    nu.setup_ax(ax, xlab, ylab, title)


def one_row(axes, cfg, M, xs, W, N, title, xlab, ylab):
    """Fill the two panels of one row. W = (Q, Q) or (U, V)."""
    beta = nu.pre_norm(M, cfg)
    Xs = nu.orbit_matrix(M, nu.coeffs_a(cfg["d"]), cfg["n_iters"], scale=beta,
                         tol=cfg.get("tol"))
    coords = [nu.spec_coords(Xs[int(k)], W[0], W[1]) for k in cfg["k_show"]]
    tgt = nu.spec_coords(N, W[0], W[1])

    ks = np.arange(cfg["n_iters"] + 1)
    errF = np.array([float(np.linalg.norm(Xs[k] - N)) for k in ks])
    err2 = np.array([float(np.linalg.norm(Xs[k] - N, 2)) for k in ks])
    resid = np.array([nu.frame_resid(Xs[k], W[0], W[1]) for k in ks])

    plot_spec(axes[0], np.asarray(xs) / beta, tgt, coords, cfg["k_show"],
              title, xlab, ylab)
    nu.plot_decay(axes[1], ks, [errF, err2],
                  [r"$\|\cdot\|_F$", r"$\|\cdot\|_2$"],
                  r"distance to $\mathrm{msgn}(M)$", floor=cfg["floor"])

    print("\n%s" % title)
    print("%-4s %-13s %-13s %-13s" % ("k", "errF", "err2", "resid"))
    for k in ks:
        print("%-4d %-13.3e %-13.3e %-13.3e" % (k, errF[k], err2[k], resid[k]))


def run(cfg=CFG):
    fig, axes = plt.subplots(3, 2, figsize=cfg["figsize"], squeeze=False)

    # Row 1: symmetric, read on the eigenvalues.
    S = nu.make_S(cfg)
    lam, Q = nu.calc_eig(S)
    tol_s = float(max(S.shape) * np.finfo(float).eps * np.max(np.abs(lam)))
    N_s = nu.op_eig(S, lambda t: np.where(np.abs(t) > tol_s, np.sign(t), 0.0))
    one_row(axes[0], cfg, S, lam, (Q, Q), N_s,
            r"symmetric, D=%d" % cfg["d"], r"$\lambda_i(M)$",
            "eigenvalue coordinate")

    # Row 2: rectangular, full rank.
    cfg_f = dict(cfg, n_zero=0)
    M = nu.make_M(cfg_f)
    U, sig, V = nu.calc_svd(M)
    one_row(axes[1], cfg_f, M, sig, (U, V), nu.sgn_svd(M),
            r"rectangular, full rank, D=%d" % cfg["d"], r"$\sigma_i(M)$",
            "singular value coordinate")

    # Row 3: rectangular, one zero singular value.
    cfg_z = dict(cfg, n_zero=1)
    M0 = nu.make_M(cfg_z)
    U0, sig0, V0 = nu.calc_svd(M0)
    print("\nrow 3: rank %d of %d, sigma_min = %.3e"
          % (int(np.sum(sig0 > nu.rank_tol(M0, sig0))), sig0.size, sig0.min()))
    one_row(axes[2], cfg_z, M0, sig0, (U0, V0), nu.sgn_svd(M0),
            r"rectangular, one $\sigma_i=0$, D=%d" % cfg["d"],
            r"$\sigma_i(M)$", "singular value coordinate")

    nu.output(fig, cfg["outdir"], cfg["fname"], cfg["dpi"], cfg["show"])


if __name__ == "__main__":
    run()
