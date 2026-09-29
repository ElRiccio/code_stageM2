# old_numpy

The original NumPy version of the code. It's separate from `ns_core`: nothing
here imports it, and it doesn't import anything from here.

- `ns_utils.py` has the shared helpers every script uses: polynomial
  coefficients and orbits, the sign and CPWL maps, and plotting/output
  helpers.
- `ns_utils_eline.py` is a copy of `ns_utils.py` with extra TODO comments.
  No script imports it.
- `exp1_orbits.py` through `exp17_selected_profiles.py` make the thesis
  figures, one script per family (orbits, convergence order, burn-in,
  clip/soft/CPWL profiles, sign errors and spectra, the forms catalogue). Each
  starts with a `CFG` dict (degrees, grid, output folder, `show`) and has a
  `run(cfg)` function.

## Running

The scripts import `ns_utils` as a sibling module, so run them from inside
this folder:

```
cd old_numpy
python exp1_orbits.py
```

Figures go to the `outdir` in each script's `CFG` (default `figures/`, created
in whatever folder you ran from).
