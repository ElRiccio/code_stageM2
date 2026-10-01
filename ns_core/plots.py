"""Matplotlib plots for the experiments: convergence curves, timing curves,
ranks and spectra.

Each plot function takes an optional `ax` (a new one is made if omitted) and
returns it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D

from ns_core import ns_iteration, orbit_tools


def _axis(ax):
    return plt.subplots(figsize=(5.5, 4))[1] if ax is None else ax


def _np(t: torch.Tensor):
    return t.detach().cpu().numpy()


def _label(labels: dict | None, key, default: str) -> str:
    return default if labels is None else labels.get(key, default)


def _colors(degrees) -> dict:
    cycle = plt.get_cmap("tab10")
    return {D: cycle(i % 10) for i, D in enumerate(sorted(degrees))}


def plot_error_curves(
    errors: dict[int, torch.Tensor],
    ax=None,
    eps: float | None = None,
    predicted: dict[int, int] | None = None,
    ylabel: str | None = None,
    labels: dict | None = None,
    show_predicted: bool = True,
    show_first_hit: bool = True,
    show_eps_line: bool = True,
    k_plot: int | None = None,
):
    """
    errors: error curve per degree
    ax: axis to draw on
    eps: target level (needed for the first-hit markers and the eps line)
    predicted: K_D per degree (needed for the predicted lines)
    ylabel: replaces the default y label
    labels: legend text per key (default "D=key")
    show_predicted: draw predicted K_D as dashed vertical lines
    show_first_hit: mark the first step with error <= eps
    show_eps_line: draw the horizontal line at eps
    k_plot: last step drawn (None = all)
    Returns: the axis
    """
    ax = _axis(ax)
    colors = _colors(errors)
    draw_predicted = show_predicted and predicted is not None
    draw_first_hit = show_first_hit and eps is not None
    draw_eps_line = show_eps_line and eps is not None
    for D in sorted(errors):
        e = errors[D] if k_plot is None else errors[D][: k_plot + 1]
        ax.semilogy(_np(torch.arange(len(e))), _np(e), "-o", ms=3, color=colors[D])
        if draw_predicted and D in predicted:
            ax.axvline(predicted[D], color=colors[D], ls="--", lw=1)
        if draw_first_hit:
            k = orbit_tools.first_hit(e, eps)
            if k is not None:
                ax.plot(k, float(e[k]), "s", ms=9, mfc="none", color=colors[D])
    handles = [Line2D([], [], color=colors[D], marker="o", ms=3, label=_label(labels, D, f"D={D}")) for D in sorted(errors)]
    if draw_eps_line:
        ax.axhline(eps, color="gray", ls=":")
    if draw_first_hit:
        handles.append(Line2D([], [], color="k", marker="s", mfc="none", ls="", label="first k with error $\\leq \\varepsilon$"))
    if draw_predicted:
        handles.append(Line2D([], [], color="k", ls="--", lw=1, label="predicted $K_D$"))
    if k_plot is not None:
        ax.set_xlim(0, k_plot)
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(ylabel if ylabel is not None else "$\\|X_k - \\mathrm{Sgn}(M)\\|_2$")
    ax.legend(handles=handles, fontsize=8)
    return ax


def plot_error_map(errors: dict[int, torch.Tensor], ax=None, floor: float | None = None):
    """
    errors: error curve per degree
    ax: axis to draw on
    floor: drop points at or below this error
    Returns: the axis
    Note: default floor is 3x the final error if the curve stagnated, else 0
    """
    ax = _axis(ax)
    colors = _colors(errors)
    for D in sorted(errors):
        e = errors[D]
        x, y = e[:-1], e[1:]
        if floor is not None:
            level = floor
        else:
            stagnated = len(e) > 1 and float(e[-1]) >= 0.5 * float(e[-2])
            level = 3.0 * float(e[-1]) if stagnated else 0.0
        keep = y > level
        x, y = x[keep], y[keep]
        ax.loglog(_np(x), _np(y), "o", ms=4, color=colors[D], label=f"D={D}")
        if x.numel():
            line = torch.logspace(torch.log10(x.min()), torch.log10(x.max()), 50, dtype=x.dtype)
            kappa = ns_iteration.asymptotic_error_constant(D)
            ax.loglog(_np(line), _np(kappa * line ** (D + 1)), "--", lw=1, color=colors[D])
    ax.set_xlabel("$e_k$")
    ax.set_ylabel("$e_{k+1}$")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="k", ls="--", lw=1, label="$\\kappa_D\\, e_k^{D+1}$"))
    ax.legend(handles=handles, fontsize=8)
    return ax


def _time_curve(ax, x, stats, color, label, ls="-"):
    """
    ax: axis to draw on
    x: x values
    stats: timing stats per x (from timing.describe)
    color, label, ls: line style
    """
    ax.loglog(x, [c["median"] for c in stats], ls, marker="o", ms=3, color=color, label=label)


def plot_time_vs_size(res: dict, device: str, cond: float, ax=None):
    """
    res: output of run_svd_timing
    device: which device's timings
    cond: which condition number
    ax: axis to draw on
    Returns: the axis (time vs size, one curve per degree plus SVD)
    """
    ax = _axis(ax)
    cfg, cells = res["cfg"], res["cells"]
    sizes = sorted(cfg.sizes)
    degrees = [] if cfg.svd_only else cfg.degrees
    colors = _colors(cfg.degrees)
    for D in sorted(degrees):
        _time_curve(ax, sizes, [cells[device, n, cond]["ns"][D]["time"] for n in sizes], colors[D], f"NS, D={D}")
    if cfg.run_svd:
        _time_curve(ax, sizes, [cells[device, n, cond]["svd"]["time"] for n in sizes], "k", "SVD", ls="--")
    ax.set_xlabel("size $n$")
    ax.set_ylabel("time (s)")
    ax.set_title(f"{device}, $\\kappa$ = {cond:g}")
    ax.legend(fontsize=8)
    return ax


def plot_time_vs_cond(res: dict, device: str, n: int, ax=None):
    """
    res: output of run_svd_timing
    device: which device's timings
    n: which size
    ax: axis to draw on
    Returns: the axis (time vs condition number, K_D labelled, plus SVD)
    """
    ax = _axis(ax)
    cfg, cells = res["cfg"], res["cells"]
    conds = sorted(cfg.conds)
    degrees = [] if cfg.svd_only else cfg.degrees
    colors = _colors(cfg.degrees)
    for D in sorted(degrees):
        _time_curve(ax, conds, [cells[device, n, s]["ns"][D]["time"] for s in conds], colors[D], f"NS, D={D}")
        for s in conds:
            ax.annotate(
                str(cells[device, n, s]["ns"][D]["K"]),
                (s, cells[device, n, s]["ns"][D]["time"]["median"]),
                textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7, color=colors[D],
            )
    if cfg.run_svd:
        _time_curve(ax, conds, [cells[device, n, s]["svd"]["time"] for s in conds], "k", "SVD", ls="--")
    ax.set_xlabel("condition number $\\kappa$")
    ax.set_ylabel("time (s)")
    ax.set_title(f"{device}, $n$ = {n} (labels: $K_D$)")
    ax.legend(fontsize=8)
    return ax


def plot_iterations_vs_cond(
    res: dict,
    ax=None,
    show_predicted: bool = True,
    show_first_hit: bool = True,
):
    """
    res: output of run_conditioning
    ax: axis to draw on
    show_predicted: draw predicted K_D as dashed lines
    show_first_hit: draw the observed first step with error <= eps
    Returns: the axis (iterations vs condition number, one curve per degree)
    Note: a point where eps was missed within k_max is drawn open at k_max
    """
    ax = _axis(ax)
    cfg = res["cfg"]
    conds = sorted(cfg.conds)
    colors = _colors(cfg.degrees)
    for D in sorted(cfg.degrees):
        if show_first_hit:
            ys = [res["iterations"][s][D] for s in conds]
            ys = [cfg.k_max if y is None else y for y in ys]
            ax.semilogx(conds, ys, "-", color=colors[D], label=f"D={D}")
            for s, y in zip(conds, ys):
                missed = res["iterations"][s][D] is None
                ax.plot(s, y, "o", ms=4, color=colors[D], mfc="none" if missed else colors[D])
        if show_predicted:
            ax.semilogx(conds, [res["predicted"][s][D] for s in conds], "--", lw=1, color=colors[D],
                        label=None if show_first_hit else f"D={D}")
    handles, _ = ax.get_legend_handles_labels()
    if show_predicted and show_first_hit:
        handles.append(Line2D([], [], color="k", ls="--", lw=1, label="predicted $K_D$"))
    ax.set_xlabel("condition number $\\kappa$")
    ax.set_ylabel("iterations to reach $\\varepsilon$")
    ax.set_title(f"$\\varepsilon$ = {res['eps']:g}")
    ax.legend(handles=handles, fontsize=8)
    return ax


def plot_time_vs_degree(
    res: dict,
    device: str,
    dtype: torch.dtype,
    n: int,
    cond: float,
    eps: float,
    ax=None,
    label: str | None = None,
    show_svd: bool = True,
):
    """
    res: output of run_time_to_accuracy (or run_cpwl_time_to_accuracy)
    device, dtype, n, cond, eps: which setting to plot
    ax: axis to draw on
    label: legend entry, lets several settings share one axis
    show_svd: draw the SVD time as a dashed line if it was timed
    Returns: the axis (median time vs degree, ±std bars, median K labelled)
    Note: the SVD line needs svd_reference_line=True in the run, otherwise nothing is drawn
    """
    ax = _axis(ax)
    degrees = sorted(res["cfg"].degrees)
    cell = res["cells"][device, dtype, n, cond, eps]
    med = [cell[D]["time"]["median"] for D in degrees]
    std = [cell[D]["time"]["std"] for D in degrees]
    line = ax.errorbar(degrees, med, yerr=std, marker="o", ms=4, capsize=3, label=label)
    for D, t in zip(degrees, med):
        ax.annotate(
            f"{cell[D]['K']['median']:g}", (D, t), textcoords="offset points", xytext=(0, 6),
            ha="center", fontsize=7, color=line[0].get_color(),
        )
    ax.set_xticks(degrees)
    ax.set_xlabel("degree $D$")
    ax.set_ylabel("time to reach $\\varepsilon$ (s)")
    name = str(dtype).removeprefix("torch.")
    ax.set_title(f"{device}, {name}, $n$ = {n}, $\\kappa$ = {cond:g}, $\\varepsilon$ = {eps:g} (labels: $K$)")
    if show_svd:
        add_svd_reference_line(
            ax, res, device, dtype, n, cond,
            label="SVD" if label is None else f"SVD ({label})",
            color="k" if label is None else line[0].get_color(),
        )
    if label is not None:
        ax.legend(fontsize=8)
    return ax


def plot_step_cost_vs_degree(res: dict, device: str, n: int, ax=None):
    """
    res: output of run_step_cost
    device: which device's timings
    n: which size
    ax: axis to draw on
    Returns: the axis (median time of one step vs degree, plain and Gram-based step, ±std bars)
    """
    ax = _axis(ax)
    degrees = sorted(res["cfg"].degrees)
    cell = res["cells"][device, n]
    for variant, label in (("matrix", "plain step"), ("gram", "Gram step")):
        ax.errorbar(
            degrees, [cell[D][variant]["median"] for D in degrees],
            yerr=[cell[D][variant]["std"] for D in degrees], marker="o", ms=4, capsize=3, label=label,
        )
    ax.set_xticks(degrees)
    ax.set_xlabel("degree $D$")
    ax.set_ylabel("time per step (s)")
    ax.set_title(f"{device}, $n$ = {n}")
    ax.legend(fontsize=8)
    return ax


def add_svd_reference_line(
    ax, res: dict, device: str, dtype: torch.dtype, n: int, cond: float, label: str = "SVD", color: str = "k"
):
    """
    ax: axis to draw on
    res: result dict with an "svd" entry (needs svd_reference_line=True)
    device, dtype, n, cond: which setting
    label: legend entry
    color: line colour
    Returns: the axis
    Note: does nothing if the setting has no SVD timing
    """
    svd = res.get("svd", {})
    key = (device, dtype, n, cond)
    if key not in svd:
        return ax
    ax.axhline(svd[key]["median"], color=color, ls="--", lw=1, label=label)
    ax.legend(fontsize=8)
    return ax


def plot_ranks(ranks: dict[int, torch.Tensor], r: int | None = None, ax=None):
    """
    ranks: rank curve per degree
    r: true rank of M, drawn as a dashed line
    ax: axis to draw on
    Returns: the axis
    """
    ax = _axis(ax)
    colors = _colors(ranks)
    for D in sorted(ranks):
        k = _np(torch.arange(len(ranks[D])))
        ax.plot(k, _np(ranks[D]), "-o", ms=3, color=colors[D], label=f"D={D}")
    if r is not None:
        ax.axhline(r, color="gray", ls="--", label=f"rank of $M$ = {r}")
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel("numerical rank of $X_k$")
    ax.legend(fontsize=8)
    return ax


def plot_spectrum(
    sigma: torch.Tensor,
    target: torch.Tensor,
    coords: dict[int, torch.Tensor],
    ax=None,
    xaxis: str = "sigma",
    labels: dict | None = None,
    ks: list[int] | None = None,
):
    """
    sigma: input singular values
    target: exact profile value at each sigma
    coords: output spectrum per iteration count k
    ax: axis to draw on
    xaxis: "sigma" (values) or "index" (rank order)
    labels: legend text per key (default "k=key")
    ks: iteration counts to draw (None = all in coords)
    Returns: the axis
    """
    if ks is not None:
        coords = {k: coords[k] for k in ks}
    ax = _axis(ax)
    order = torch.argsort(sigma)
    s = sigma[order]
    if xaxis == "sigma":
        x, xlabel = _np(s), "$\\sigma_i(M)$"
    elif xaxis == "index":
        x, xlabel = _np(torch.arange(1, s.numel() + 1)), "index $i$ (increasing $\\sigma$)"
    else:
        raise ValueError("xaxis must be 'sigma' or 'index'")
    ax.plot(x, _np(s), ":", color="0.35", lw=1.2, label="$\\sigma_i(M)$")
    colors = _colors(coords)
    for k in sorted(coords):
        ax.plot(x, _np(coords[k][order]), lw=1.0, color=colors[k], label=_label(labels, k, f"k={k}"))
    ax.plot(x, _np(target[order]), "k--", lw=1.3, label="exact (SVD)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("spectral coordinate")
    ax.legend(fontsize=8)
    return ax
