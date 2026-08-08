# exp9_soft_spectrum.py
"""Experiment 9: the nuclear-norm proximal operator through Sgn alone.

Validates Thm. soft and Cor. nuclear-svdfree. Two independent references: the
spectral one applies soft_gamma to the singular values through an SVD, and the
variational one probes the objective gamma ||Z||_* + (1/2)||Z-M||_F^2 around the
computed point, which must not decrease in any direction.

Left panel:  sigma_i(M) dotted, soft_gamma(sigma_i) dashed, NS values at a few k.
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
    "normalize": True,       # ||M||_2 = 1, so gamma scales with sigma
    "gamma": 0.25,
    "d": 3,
    "k_show": (1, 2, 3, 5),
    "k_err": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
    "norm_method": "svd",
    "norm_power_iters": 300,
    "norm_tol": 1e-12,
    "norm_pad": 1.0 + 1e-6,
    "n_probe": 24,           # variational check
    "probe_step": 1e-3,
    "floor": 1e-18,
    "figsize": (11.0, 4.2),
    "outdir": "figures",
    "fname": "exp9_soft_spectrum.png",
    "dpi": 150,
    "show": True,
}


def prox_obj(Z, M, gamma):
    """gamma ||Z||_* + (1/2) ||Z-M||_F^2, the objective prox minimizes."""
    return (gamma * float(np.sum(nu.svdvals(Z)))
            + 0.5 * float(np.linalg.norm(Z - M)) ** 2)


def prox_gap(cfg, M, Y):
    """Smallest objective increase over random probes around Y; must be >= 0."""
    rng = np.random.default_rng(cfg["seed"] + 1)
    base = prox_obj(Y, M, cfg["gamma"])
    gaps = []
    for _ in range(int(cfg["n_probe"])):
        D = rng.standard_normal(np.shape(M))
        nrm = float(np.linalg.norm(D))
        if nrm == 0.0:
            continue
        D = (cfg["probe_step"] / nrm) * D
        gaps.append(min(prox_obj(Y + D, M, cfg["gamma"]),
                        prox_obj(Y - D, M, cfg["gamma"])) - base)
    return float(np.min(gaps)) if gaps else float("nan")


def run(cfg=CFG):
    M = nu.make_M(cfg)
    U, sig, V = nu.calc_svd(M)
    f = lambda t: nu.soft_g(t, cfg["gamma"])  # already odd, hence its own odd extension
    Y_ex = nu.op_svd(M, f)
    coords_ex = np.asarray(f(sig))

    k_all = sorted(set(int(k) for k in cfg["k_show"]) | set(int(k) for k in cfg["k_err"]))
    res = nu.sweep_op(lambda k: nu.soft_map(M, cfg["gamma"], nu.sgn_cfg(cfg, k)),
                      k_all, Y_ex, U, V)

    pos = sig[sig > nu.rank_tol(M, sig)]
    print("gen=%s, gamma=%.3f, d=%d, rank %d -> %d, gap to gamma %.3e"
          % (cfg["gen"], cfg["gamma"], cfg["d"], pos.size,
             int(np.sum(coords_ex > 0.0)), float(np.min(np.abs(pos - cfg["gamma"])))))
    print("variational check, min objective increase: %.3e (>= 0 expected)"
          % prox_gap(cfg, M, Y_ex))
    print("%-4s %-13s %-13s %-13s" % ("k", "errF", "err2", "frame resid"))
    for k in cfg["k_err"]:
        r = res[int(k)]
        print("%-4d %-13.3e %-13.3e %-13.3e" % (int(k), r["errF"], r["err2"], r["resid"]))

    fig, axes = plt.subplots(1, 2, figsize=cfg["figsize"])
    nu.plot_spectrum(axes[0], sig, coords_ex,
                     [res[int(k)]["coords"] for k in cfg["k_show"]], cfg["k_show"],
                     r"nuclear prox, $\gamma=%.2f$, d=%d" % (cfg["gamma"], cfg["d"]))
    axes[0].axhline(0.0, color="0.7", lw=0.6)
    ks = [int(k) for k in k_all]
    nu.plot_decay(axes[1], ks,
                  [[res[k]["errF"] for k in ks], [res[k]["err2"] for k in ks]],
                  [r"$\|\cdot\|_F$", r"$\|\cdot\|_2$"],
                  "distance to the exact operator", floor=cfg["floor"])
    nu.output(fig, cfg["outdir"], cfg["fname"], cfg["dpi"], cfg["show"])


if __name__ == "__main__":
    run()
