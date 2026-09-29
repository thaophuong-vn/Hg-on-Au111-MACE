# Hg/Au(111) validation run

This folder contains the validation script and the concise results transcribed from the successful WSL run `run_20260929_163121_312267`.

## Model and setup

- ASE Au(111), 4 atomic layers, lateral cells 3x3, 4x4, and 5x5.
- MACE-MP medium, float64, CPU, with D3(BJ/PBE).
- Periodic in-plane (`PBC = [true, true, false]`), 10 Å vacuum margin on each side.
- BFGS with `fmax = 0.003 eV/Å`; clean slab and adsorbed slab use matching constraints.
- One-factor-at-a-time checks: lateral cell size; 0/1/2 fixed bottom layers; Hg starting heights 1.8/2.0/2.2/2.5 Å.

## Result summary

All 32 distinct adsorption relaxations and the five matching clean-slab references reported as converged. The clean-slab energies and force maxima (rounded as printed) were:

| Reference | Energy (eV) | Free-atom fmax (eV/Å) |
|---|---:|---:|
| 3x3x4, 0 fixed | -129.2341673 | 0.0029 |
| 4x4x4, 0 fixed | -229.7496307 | 0.0029 |
| 5x5x4, 0 fixed | -358.9837980 | 0.0029 |
| 4x4x4, 1 fixed | -229.7496976 | 0.0010 |
| 4x4x4, 2 fixed | -229.3327956 | 0.0021 |

The fcc and hcp adsorption-energy gap stayed close to 5 meV across cell size, bottom-layer constraints, and initial height. Bridge starts relaxed into the fcc basin in every reported case, so these runs do not establish a distinct bridge minimum. The ontop adsorption energy was about 0.19 eV less favorable than fcc. See `validation_summary.csv` for the reported site energies.

The CSV contains values rounded to the precision printed in the WSL terminal output. The exact run folder, trajectories, structures, logs, full-precision summary CSV, and settings JSON remain in the user's WSL directory; they were not available in this upload staging environment.
