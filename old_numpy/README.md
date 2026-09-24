# old_numpy

The original NumPy implementation. Independent of `ns_core`: nothing in
either package imports the other.

- `ns_utils.py` — shared helpers (polynomial coefficients and orbits, sign
  and CPWL maps, plotting/output helpers) used by every script.
- `ns_utils_eline.py` — copy of `ns_utils.py` with extra TODO comments; not
  imported by any script.
- `exp1_orbits.py` … `exp17_selected_profiles.py` — one script per figure
  family (orbits, convergence order, burn-in, clip/soft/CPWL profiles,
  sign-error and spectrum plots, forms catalogue). Each has a `CFG` dict at
  the top (degrees, grid, output folder, `show`) and a `run(cfg)` function.

## Running

Scripts import `ns_utils` as a sibling module, so run them from inside this
folder:

```
cd old_numpy
python exp1_orbits.py
```

Figures are written to the `outdir` set in each script's `CFG` (default
`figures/`, created relative to the working directory).
