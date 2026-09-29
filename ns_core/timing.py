"""Timing helpers: time one call (with warm-up and CUDA sync) and summarize
repeated timings."""

from __future__ import annotations

import time
from typing import Callable

import torch


def time_call(
    fn: Callable[[], torch.Tensor],
    device: torch.device | str,
    n_warmup: int = 0,
) -> tuple[float, torch.Tensor]:
    """
    fn: function to time (no arguments)
    device: where fn runs
    n_warmup: untimed calls first
    Returns: seconds, output of the timed call
    Note: syncs CUDA before and after
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
    """
    values: timings
    Returns: dict with median, mean, std
    """
    t = torch.tensor(values, dtype=torch.float64)
    std = float(t.std()) if t.numel() > 1 else 0.0
    return {"median": float(torch.quantile(t, 0.5)), "mean": float(t.mean()), "std": std}
