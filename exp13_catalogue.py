# exp13_catalogue.py
"""Experiment 13: the catalogue of Tab. forms, read on both sides.

run_sym()       Sym^n, five profiles, eigenvalues reached and operator error.
run_rect()      R^{m x n}, three profiles, singular values reached and error.
run_rect_zero() R^{m x n} with one zero singular value, clip with alpha > 0.

Column 1 shows the profile: as it stands on Sym^n, on the positive half-line
only on R^{m x n} (Cor. pos-only: the sign form sees nothing else), and as its
odd extension in the last figure, where the origin is the point at issue.
"""

import numpy as np
import matplotlib.pyplot as plt

import ns_utils as nu

CFG = {
    "m": 64,
    "n": 48,
    "seed": 0,
    "gen": "edit",
    "n_zero": 0,
    "n_small": 0,
    "s_small": 1e-6,
    "normalize": True,        # ||M||_2 = 1, so every knot below scales with it
    "d": 3,              # displayed as D; the key stays "d" for nu.sgn_cfg
    "k_show": (1, 2, 3, 5),
    "k_err": tuple(range(1, 17)),
    "xaxis": "sigma",
    "norm_method": "svd",
    "norm_power_iters": 300,
    "norm_tol": 1e-12,
    "norm_pad": 1.0 + 1e-6,
    "floor": 1e-18,
    "n_grid": 1201,
    "outdir": "figures",
    "dpi": 150,
    "show": True,
    "tol": 1e-12,
}

# Parameters are read against ||M||_2 = 1, so every knot sits in the spectrum.
ALPHA, BETA = -0.30, 0.60     # clip on Sym^n
ALPHA_P = 0.20                # clip on R^{m x n}, the starred case alpha > 0
GAMMA = 0.25                  # soft
A_LEAK = 0.20                 # leak slope
BETA_C = 0.55                 # cap of the capped leaky rectifier
MU_LKC = 0.40                 # knot of the leaky clipping profile

SYM = [
    (r"$\mathrm{clip}_{[%.2f,\,%.2f]}$" % (ALPHA, BETA),
     lambda x: nu.clip_ab(x, ALPHA, BETA),
     lambda M, S: nu.clip_map_sym(M, ALPHA, BETA, S)),
    (r"$\mathrm{soft}_{%.2f}$" % GAMMA,
     lambda x: nu.soft_g(x, GAMMA),
     lambda M, S: nu.soft_map_sym(M, GAMMA, S)),
    (r"$\mathrm{lrelu}_{%.2f}$" % A_LEAK,
     lambda x: nu.lrelu(x, A_LEAK),
     lambda M, S: nu.lrelu_map_sym(M, A_LEAK, S)),
    (r"$\mathrm{crelu}_{%.2f,\,%.2f}$" % (A_LEAK, BETA_C),
     lambda x: nu.crelu(x, A_LEAK, BETA_C),
     lambda M, S: nu.crelu_map_sym(M, A_LEAK, BETA_C, S)),
    (r"$\mathrm{lkc}_{%.2f,\,%.2f}$" % (A_LEAK, MU_LKC),
     lambda x: nu.lkc(x, A_LEAK, MU_LKC),
     lambda M, S: nu.lkc_map_sym(M, A_LEAK, MU_LKC, S)),
]

RECT = [
    (r"$\mathrm{clip}_{[%.2f,\,%.2f]}$" % (ALPHA_P, BETA),
     lambda x: nu.clip_ab(x, ALPHA_P, BETA),
     lambda M, S: nu.clip_map(M, ALPHA_P, BETA, S)),
    (r"$\mathrm{soft}_{%.2f}$" % GAMMA,
     lambda x: nu.soft_g(x, GAMMA),
     lambda M, S: nu.soft_map(M, GAMMA, S)),
    (r"$\mathrm{crelu}_{%.2f,\,%.2f}$" % (A_LEAK, BETA_C),
     lambda x: nu.crelu(x, A_LEAK, BETA_C),
     lambda M, S: nu.crelu_map(M, A_LEAK, BETA_C, S)),
]


def odd_ext_tol(h, tol):
    """Odd extension with sgn(t) = 0 for |t| <= tol, the convention of sgn_svd."""

    def g(x):
        x = np.asarray(x, dtype=float)
        s = np.where(np.abs(x) > tol, np.sign(x), 0.0)
        return s * np.asarray(h(np.abs(x)), dtype=float)

    return g


def share_limits(axs, x_prof, series):
    """Put the profile panel and the spectrum panel on one pair of limits."""
    x_prof = np.asarray(x_prof, dtype=float)
    xlo, xhi = float(x_prof.min()), float(x_prof.max())
    v = np.concatenate([np.asarray(a, dtype=float).ravel() for a in series])
    ylo, yhi = float(v.min()), float(v.max())
    pad = 0.05 * max(xhi - xlo, yhi - ylo)
    for ax in axs:
        ax.set_xlim(xlo - pad, xhi + pad)
        ax.set_ylim(ylo - pad, yhi + pad)


def plot_spec(ax, xs, exact, approx, ks, title, xlab, ylab):
    """Input spectrum (dotted), target (dashed) and the values at a few k."""
    xs = np.asarray(xs, dtype=float)
    order = np.argsort(xs)
    x = xs[order]
    ax.plot(x, x, ":", color="0.35", lw=1.2, label="input")
    for k, v in zip(ks, approx):
        ax.plot(x, np.asarray(v, dtype=float)[order], lw=1.0, label="k=%d" % int(k))
    ax.plot(x, np.asarray(exact, dtype=float)[order], "k--", lw=1.3, label="exact")
    nu.setup_ax(ax, xlab, ylab, title)


def one_row(axes, cfg, M, xs, W, rows_spec, ref_f, x_prof, xlab, ylab, prof_title):
    """Fill the three panels of one row. W is the frame, U = V = Q if symmetric."""
    label, h, F = rows_spec
    Y_ex = (nu.op_eig(M, ref_f) if ylab.startswith("eigen")
            else nu.op_svd(M, ref_f))
    coords_ex = np.asarray(ref_f(xs))

    k_all = sorted(set(int(k) for k in cfg["k_show"])
                   | set(int(k) for k in cfg["k_err"]))
    res = nu.sweep_op(lambda k: F(M, nu.sgn_cfg(cfg, k)), k_all, Y_ex, W[0], W[1])

    ax = axes[0]
    ax.plot(x_prof, h(x_prof), "-", lw=1.4, color="C0")
    ax.axhline(0.0, color="0.85", lw=0.5)
    ax.axvline(0.0, color="0.85", lw=0.5)
    nu.setup_ax(ax, "x", label, prof_title, legend=False)

    plot_spec(axes[1], xs, coords_ex, [res[int(k)]["coords"] for k in cfg["k_show"]],
              cfg["k_show"], "spectral coordinates, D=%d" % cfg["d"], xlab, ylab)

    share_limits(axes[:2], x_prof, [h(x_prof), coords_ex, xs])

    ks = [int(k) for k in k_all]
    nu.plot_decay(axes[2], ks,
                  [[res[k]["errF"] for k in ks], [res[k]["err2"] for k in ks]],
                  [r"$\|\cdot\|_F$", r"$\|\cdot\|_2$"],
                  "distance to the exact operator", floor=cfg["floor"])
    return res


def report(name, res, k_err):
    print("\n%s" % name)
    print("%-4s %-13s %-13s %-13s" % ("k", "errF", "err2", "resid"))
    for k in k_err:
        r = res[int(k)]
        print("%-4d %-13.3e %-13.3e %-13.3e"
              % (int(k), r["errF"], r["err2"], r["resid"]))


def run_sym(cfg=CFG):
    """Figure 1: Sym^n, five profiles, no odd extension anywhere."""
    S = nu.make_S(cfg)
    lam, Q = nu.calc_eig(S)
    print("Sym^%d, D=%d, lambda in [%.3f, %.3f], min |lambda| = %.3e"
          % (cfg["n"], cfg["d"], lam.min(), lam.max(), np.min(np.abs(lam))))

    x_prof = np.linspace(-1.0, 1.0, cfg["n_grid"])
    fig, axes = plt.subplots(len(SYM), 3, figsize=(13.5, 3.4 * len(SYM)),
                             squeeze=False)
    for i, spec in enumerate(SYM):
        res = one_row(axes[i], cfg, S, lam, (Q, Q), spec, spec[1], x_prof,
                      r"$\lambda_i(M)$", "eigenvalue coordinate", "profile")
        report(spec[0], res, cfg["k_err"])
    nu.output(fig, cfg["outdir"], "exp13_sym.png", cfg["dpi"], cfg["show"])


def run_rect(cfg=CFG):
    """Figure 2: R^{m x n}, three profiles, profile drawn on [0, 1.2] only."""
    M = nu.make_M(cfg)
    U, sig, V = nu.calc_svd(M)
    f_tol = nu.rank_tol(M, sig)
    print("R^{%dx%d}, D=%d, rank %d of %d, sigma_min = %.3e"
          % (cfg["m"], cfg["n"], cfg["d"],
             int(np.sum(sig > f_tol)), sig.size, sig.min()))

    x_prof = np.linspace(0.0, 1.0, cfg["n_grid"])
    fig, axes = plt.subplots(len(RECT), 3, figsize=(13.5, 3.4 * len(RECT)),
                             squeeze=False)
    for i, spec in enumerate(RECT):
        res = one_row(axes[i], cfg, M, sig, (U, V), spec,
                      odd_ext_tol(spec[1], f_tol), x_prof,
                      r"$\sigma_i(M)$", "singular value coordinate",
                      r"profile on $[0,\infty)$")
        report(spec[0], res, cfg["k_err"])
    nu.output(fig, cfg["outdir"], "exp13_rect.png", cfg["dpi"], cfg["show"])


def run_rect_zero(cfg=CFG):
    """Figure 3: one zero singular value, clip with alpha > 0, odd extension."""
    cfg = dict(cfg)
    cfg["n_zero"] = 1
    M = nu.make_M(cfg)
    U, sig, V = nu.calc_svd(M)
    tol = nu.rank_tol(M, sig)
    h = lambda x: nu.clip_ab(x, ALPHA_P, BETA)
    f = odd_ext_tol(h, tol)
    print("R^{%dx%d}, D=%d, rank %d of %d, alpha=%.2f > 0"
          % (cfg["m"], cfg["n"], cfg["d"], int(np.sum(sig > tol)), sig.size,
             ALPHA_P))

    Y_ex = nu.op_svd(M, f)
    coords_ex = np.asarray(f(sig))
    k_all = sorted(set(int(k) for k in cfg["k_show"])
                   | set(int(k) for k in cfg["k_err"]))
    res = nu.sweep_op(lambda k: nu.clip_map(M, ALPHA_P, BETA, nu.sgn_cfg(cfg, k)),
                      k_all, Y_ex, U, V)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), squeeze=False)
    ax = axes[0][0]
    xp = np.linspace(1e-3, 1.0, cfg["n_grid"])
    ax.plot(xp, f(xp), "-", lw=1.4, color="C0")
    ax.plot(-xp, f(-xp), "-", lw=1.4, color="C0")
    ax.plot([0.0], [0.0], "o", ms=5, color="C3", label=r"$\hat{h}(0)=0$")
    ax.plot([0.0, 0.0], [-ALPHA_P, ALPHA_P], ":", lw=1.0, color="C3")
    ax.axhline(0.0, color="0.85", lw=0.5)
    ax.axvline(0.0, color="0.85", lw=0.5)
    nu.setup_ax(ax, "x", r"$\hat{h}=\mathrm{odd\ ext.}$ of $\mathrm{clip}_{[%.2f,%.2f]}$"
                % (ALPHA_P, BETA), "odd extension, discontinuous at 0")

    plot_spec(axes[0][1], sig, coords_ex,
              [res[int(k)]["coords"] for k in cfg["k_show"]], cfg["k_show"],
              "spectral coordinates, D=%d" % cfg["d"],
              r"$\sigma_i(M)$", "singular value coordinate")
    share_limits(axes[0][:2], np.concatenate([-xp, xp]),
                 [f(xp), f(-xp), coords_ex, sig])

    ks = [int(k) for k in k_all]
    nu.plot_decay(axes[0][2], ks,
                  [[res[k]["errF"] for k in ks], [res[k]["err2"] for k in ks]],
                  [r"$\|\cdot\|_F$", r"$\|\cdot\|_2$"],
                  "distance to the exact operator", floor=cfg["floor"])
    report("clip, alpha>0, one zero singular value", res, cfg["k_err"])
    nu.output(fig, cfg["outdir"], "exp13_rect_zero.png", cfg["dpi"], cfg["show"])


if __name__ == "__main__":
    run_sym()
    run_rect()
    run_rect_zero()
