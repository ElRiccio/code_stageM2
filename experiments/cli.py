"""Shared --device/--dtype command-line option for every driver in
experiments/scripts/, so any experiment can be pointed at CPU or CUDA (and
its dtype) from the command line instead of editing constants in the
script (reviewer: "possibly GPU versus CPU performance if available").

Kept to two small argparse-based helpers, not a general CLI framework, to
match the project's "keep it simple" convention for a 2-3 person codebase.
"""

from __future__ import annotations

import argparse

import torch

_DTYPES = {"float32": torch.float32, "float64": torch.float64}
_DTYPE_NAMES = {torch.float32: "float32", torch.float64: "float64"}


def dtype_name(dtype: torch.dtype) -> str:
    """"float32"/"float64" for a torch dtype: the short form used in this
    suite's filenames, titles and CSV columns (str(dtype) itself reads as
    "torch.float32", which is noisier than needed there)."""
    return _DTYPE_NAMES[dtype]


def _add_dtype_arg(parser: argparse.ArgumentParser, default_dtype: torch.dtype) -> None:
    parser.add_argument(
        "--dtype", choices=list(_DTYPES), default=_DTYPE_NAMES[default_dtype],
        help=f"floating-point dtype (default: {_DTYPE_NAMES[default_dtype]})",
    )


def parse_device_dtype(
    *, default_dtype: torch.dtype = torch.float64, description: str = ""
) -> tuple[torch.device, torch.dtype]:
    """Parses `--device {auto,cpu,cuda}` (default "auto": cuda if available,
    else cpu) and `--dtype {float32,float64}`. Returns (device, dtype),
    ready to pass straight through to an experiment function's `device`/
    `dtype` kwargs. Exits with a clear message if --device cuda is
    requested but no CUDA device is present, rather than failing deep
    inside torch with a less legible error.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="device to run on (default: auto = cuda if available, else cpu)",
    )
    _add_dtype_arg(parser, default_dtype)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but no CUDA device is available.")
        device = torch.device(args.device)
    return device, _DTYPES[args.dtype]


def parse_devices_dtype(
    *, default_dtype: torch.dtype = torch.float32, description: str = ""
) -> tuple[list[str], torch.dtype]:
    """Like `parse_device_dtype`, but for a driver (svd_device.py) whose
    whole point is comparing devices: `--device {both,cpu,cuda}` (default
    "both") selects which device(s) to run, returned as a list ready to
    pass as the experiment function's `devices` kwarg. "both" with no CUDA
    present still returns just ["cpu"] (the experiment function itself
    already skips unavailable devices; --device cuda with no CUDA present
    still exits with a clear message, as in `parse_device_dtype`).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--device", choices=["both", "cpu", "cuda"], default="both",
        help="device(s) to run on (default: both = cpu and cuda, if available)",
    )
    _add_dtype_arg(parser, default_dtype)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA device is available.")
    devices = ["cpu", "cuda"] if args.device == "both" else [args.device]
    return devices, _DTYPES[args.dtype]
