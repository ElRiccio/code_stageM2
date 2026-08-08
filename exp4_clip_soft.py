# exp4_clip_soft.py
"""Experiment 4: clip and soft built from the iterated sign, several NS counts."""

import numpy as np
import matplotlib.pyplot as plt

import ns_utils as nu

CFG = {
    "d": 3,                          # degree of the inner iteration
    "ns_list": (1, 2, 3, 5, 8),      # number of Newton-Schulz steps
    "alpha": -0.4,
    "beta": 0.8,
    "gamma": 0.5,
    "x_lim": (-1.5, 1.5),
    "n_grid": 1201,
    "figsize": (10.0, 4.2),
    "outdir": "figures",
    "dpi": 150,
    "show": True,
}


def run(cfg=CFG):
    x = np.linspace(cfg["x_lim"][0], cfg["x_lim"][1], cfg["n_grid"])
    clip_ref = nu.clip_ab(x, cfg["alpha"], cfg["beta"])
    soft_ref = nu.soft_g(x, cfg["gamma"])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=cfg["figsize"])
    print("%4s %14s %14s" % ("ns", "sup err clip", "sup err soft"))
    for ns in cfg["ns_list"]:
        sgn = nu.make_sgn(cfg["d"], ns)  # rescales each sign argument on its own
        c = nu.clip_sgn(x, cfg["alpha"], cfg["beta"], sgn)
        s = nu.soft_sgn(x, cfg["gamma"], sgn)
        print("%4d %14.3e %14.3e" % (ns, np.max(np.abs(c - clip_ref)),
                                     np.max(np.abs(s - soft_ref))))
        ax0.plot(x, c, lw=1.0, label="ns=%d" % ns)
        ax1.plot(x, s, lw=1.0, label="ns=%d" % ns)

    ax0.plot(x, clip_ref, "k--", lw=1.0, label="exact")
    ax1.plot(x, soft_ref, "k--", lw=1.0, label="exact")
    nu.setup_ax(ax0, "x", r"$\mathrm{clip}_{[\alpha,\beta]}(x)$",
                "clip, alpha=%.2f, beta=%.2f (d=%d)" % (cfg["alpha"], cfg["beta"], cfg["d"]))
    nu.setup_ax(ax1, "x", r"$\mathrm{soft}_\gamma(x)$",
                "soft, gamma=%.2f (d=%d)" % (cfg["gamma"], cfg["d"]))
    nu.output(fig, cfg["outdir"], "exp4_clip_soft.png", cfg["dpi"], cfg["show"])


if __name__ == "__main__":
    run()