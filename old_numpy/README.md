# old_numpy

The original NumPy version. It is independent of `ns_core`.

- `ns_utils.py`: shared helpers (polynomials, orbits, sign and CPWL maps,
  plotting).
- `ns_utils_eline.py`: copy of `ns_utils.py` with extra TODO comments. Unused.
- `exp1_orbits.py` to `exp17_selected_profiles.py`: one script per figure
  family. Each has a `CFG` dict and a `run(cfg)` function.

Run from inside this folder, since the scripts import `ns_utils`:

```
cd old_numpy
python exp1_orbits.py
```

Figures go to the `outdir` set in `CFG` (default `figures/`).
