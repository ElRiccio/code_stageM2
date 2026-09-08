# exp15_orbit_radius.py
"""Experiment 15: iterates of p_D past the radius R_D, for D = 1..6.

The same picture as exp1, drawn on an interval wider than [-1,1]: each panel
shows p_D^{ok}(x) for k = 0,...,n_iters against the target sgn. Inside
(-R_D, R_D) the iterates flatten onto +-1 (Prop. basin (ii)); outside they leave
the frame, and R_D is the exact place where that starts (Thm. radius (iii)).
"""

import numpy as np
import matplotlib.pyplot as plt

import ns_utils as nu

CFG = {
    "D_list": (1, 2, 3, 4, 5, 6),
    "n_grid": 2001,
    "n_iters": 5,
    "x_pad": 1.22,          # panels run over (-x_pad R_D, x_pad R_D)
    "y_lim": (-2.2, 2.2),
    "figsize": (14.0, 7.6),
    "outdir": "figures",
    "fname": "exp15_orbit_radius.png",
    "dpi": 150,
    "show": True,
}


def run(cfg=CFG):
    fig, axes = plt.subplots(2, 3, figsize=cfg["figsize"], squeeze=False)
    print("%-3s %-11s %-11s %-s" % ("D", "R_D", "p_D(R_D)", "boundary orbit"))

    for i, D in enumerate(cfg["D_list"]):
        ax = axes[i // 3][i % 3]
        R, _ = nu.radius_D(D)
        xm = cfg["x_pad"] * R
        x = np.linspace(-xm, xm, cfg["n_grid"])

        with np.errstate(over="ignore", invalid="ignore"):
            U = nu.orbit_pd(x, D, cfg["n_iters"])   # rows are the iterates

        ax.plot(x, nu.sgn_exact(x), "k--", lw=1.1, label="sgn")
        for k in range(1, cfg["n_iters"] + 1):   # k = 0 is the identity
            ax.plot(x, U[k], lw=1.0, label="k=%d" % k)
        for s in (-1.0, 1.0):
            ax.axvline(s * R, color="C3", lw=1.2,
                       label=r"$\pm R_D$" if s > 0 else None)
        ax.axhline(0.0, color="0.85", lw=0.5)

        ax.set_xlim(-xm, xm)
        ax.set_ylim(cfg["y_lim"])
        nu.setup_ax(ax, "x", r"$p_D^{\circ k}(x)$",
                    r"D=%d, degree %d,  $R_D=%.4f$" % (D, 2 * D + 1, R))

        pR = float(nu.p_d(R, D))
        print("%-3d %-11.6f %-11.3e %s"
              % (D, R, pR, "fixed at R_D" if D % 2 == 0 else "sent to 0"))

    nu.output(fig, cfg["outdir"], cfg["fname"], cfg["dpi"], cfg["show"])


if __name__ == "__main__":
    run()
