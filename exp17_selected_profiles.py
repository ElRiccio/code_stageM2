# exp17_selected_profiles.py
"""Experiment 17: four single-row figures pulled out of the Tab. forms catalogue.

Each run_*() draws one 1x3 figure (profile, spectral coordinates, operator
error) for exactly one profile, reusing the row machinery of exp13_catalogue:

run_clip_eig()  clip on Sym^n (eigenvalues)
run_relu_eig()  relu on Sym^n (eigenvalues), read as lrelu with leak a=0
run_soft_svd()  soft on R^{m x n} (singular values)
run_lkc_svd()   leaky clipping on R^{m x n} (singular values)
"""

import numpy as np
import matplotlib.pyplot as plt

import ns_utils as nu
import exp13_catalogue as ec

CLIP_EIG = (r"$\mathrm{clip}_{[%.2f,\,%.2f]}$" % (ec.ALPHA, ec.BETA),
            lambda x: nu.clip_ab(x, ec.ALPHA, ec.BETA),
            lambda M, S: nu.clip_map_sym(M, ec.ALPHA, ec.BETA, S))

RELU_EIG = (r"$\mathrm{relu}$",
            lambda x: nu.relu(x),
            lambda M, S: nu.lrelu_map_sym(M, 0.0, S))

SOFT_SVD = (r"$\mathrm{soft}_{%.2f}$" % ec.GAMMA,
            lambda x: nu.soft_g(x, ec.GAMMA),
            lambda M, S: nu.soft_map(M, ec.GAMMA, S))

LKC_SVD = (r"$\mathrm{lkc}_{%.2f,\,%.2f}$" % (ec.A_LEAK, ec.MU_LKC),
           lambda x: nu.lkc(x, ec.A_LEAK, ec.MU_LKC),
           lambda M, S: nu.lkc_map(M, ec.A_LEAK, ec.MU_LKC, S))


def run_sym_one(cfg, spec, fname):
    """One 1x3 figure on Sym^n: profile as it stands, no odd extension."""
    S = nu.make_S(cfg)
    lam, Q = nu.calc_eig(S)
    print("Sym^%d, D=%d, lambda in [%.3f, %.3f], min |lambda| = %.3e"
          % (cfg["n"], cfg["d"], lam.min(), lam.max(), min(abs(lam.min()), abs(lam.max()))))

    x_prof = np.linspace(-1.0, 1.0, cfg["n_grid"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    res = ec.one_row(axes, cfg, S, lam, (Q, Q), spec, spec[1], x_prof,
                     r"$\lambda_i(M)$", "eigenvalue coordinate", "profile")
    ec.report(spec[0], res, cfg["k_err"])
    nu.output(fig, cfg["outdir"], fname, cfg["dpi"], cfg["show"])


def run_rect_one(cfg, spec, fname):
    """One 1x3 figure on R^{m x n}: profile on [0, 1] only, odd extension for the exact op."""
    M = nu.make_M(cfg)
    U, sig, V = nu.calc_svd(M)
    f_tol = nu.rank_tol(M, sig)
    print("R^{%dx%d}, D=%d, rank %d of %d, sigma_min = %.3e"
          % (cfg["m"], cfg["n"], cfg["d"], int((sig > f_tol).sum()), sig.size, sig.min()))

    x_prof = np.linspace(0.0, 1.0, cfg["n_grid"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    res = ec.one_row(axes, cfg, M, sig, (U, V), spec,
                     ec.odd_ext_tol(spec[1], f_tol), x_prof,
                     r"$\sigma_i(M)$", "singular value coordinate",
                     r"profile on $[0,\infty)$")
    ec.report(spec[0], res, cfg["k_err"])
    nu.output(fig, cfg["outdir"], fname, cfg["dpi"], cfg["show"])


def run_clip_eig(cfg=ec.CFG):
    run_sym_one(cfg, CLIP_EIG, "exp17_clip_eig.png")


def run_relu_eig(cfg=ec.CFG):
    run_sym_one(cfg, RELU_EIG, "exp17_relu_eig.png")


def run_soft_svd(cfg=ec.CFG):
    run_rect_one(cfg, SOFT_SVD, "exp17_soft_svd.png")


def run_lkc_svd(cfg=ec.CFG):
    run_rect_one(cfg, LKC_SVD, "exp17_lkc_svd.png")


if __name__ == "__main__":
    run_clip_eig()
    run_relu_eig()
    run_soft_svd()
    run_lkc_svd()
