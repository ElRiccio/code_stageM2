"""Wall-clock timing of a callable, with warm-up and CUDA synchronization, and
summary statistics of repeated measurements."""

from __future__ import annotations

import time
from typing import Callable

import torch


def time_call(
    fn: Callable[[], torch.Tensor],
    device: torch.device | str,
    n_warmup: int = 0,
) -> tuple[float, torch.Tensor]:
    """Time in seconds of one call of the zero-argument callable `fn`, after
    `n_warmup` untimed calls. On a CUDA device the stream is synchronized
    before and after the timed call, so the time is that of completed work.
    Returns (seconds, output of the timed call).

    Usage: t, X = time_call(lambda: sgn_svd(M), device="cuda", n_warmup=3)
    """
    cuda = torch.device(device).type == "cuda"
    for _ in range(n_warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if cuda:
        torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def describe(values: list[float]) -> dict[str, float]:
    """Median, mean and sample standard deviation (0 for a single value) of
    a list of measurements.

    Usage: describe([0.011, 0.012, 0.010])  # {"median": ..., "mean": ..., "std": ...}
    """
    t = torch.tensor(values, dtype=torch.float64)
    std = float(t.std()) if t.numel() > 1 else 0.0
    return {"median": float(torch.quantile(t, 0.5)), "mean": float(t.mean()), "std": std}
