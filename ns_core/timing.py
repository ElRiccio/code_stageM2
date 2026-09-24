"""Wall-clock timing of a callable, with warm-up and CUDA synchronization."""

from __future__ import annotations

import time
from typing import Callable

import torch


def time_call(
    fn: Callable[[], torch.Tensor],
    device: torch.device | str,
    n_warmup: int = 3,
    n_reps: int = 20,
) -> tuple[dict[str, float], torch.Tensor]:
    """Times the zero-argument callable `fn` (seconds): `n_warmup` untimed
    calls, then `n_reps` timed ones. On a CUDA device the stream is
    synchronized before and after every timed call, so the time is that of
    completed work. Returns ({"median", "q25", "q75"}, output of the last
    call).

    Usage: stats, X = time_call(lambda: sgn_svd(M), device="cuda", n_reps=10)
    """
    cuda = torch.device(device).type == "cuda"
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_reps):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    t = torch.tensor(times, dtype=torch.float64)
    q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    return {"median": float(q[1]), "q25": float(q[0]), "q75": float(q[2])}, out
