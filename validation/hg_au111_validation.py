#!/usr/bin/env python3
"""Compact validation suite for Hg adsorption on Au(111) with MACE-MP + D3."""

from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import csv
import json

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, fcc111
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.io import write
from ase.optimize import BFGS
from mace.calculators import mace_mp


# One-factor-at-a-time checks; duplicate cases are run only once.
SIZES = ((3, 3, 4), (4, 4, 4), (5, 5, 4))
FIXED_BOTTOM_LAYERS = (0, 1, 2)
HG_HEIGHTS = (1.8, 2.0, 2.2, 2.5)  # Angstrom above top Au layer
SITES = ("fcc", "hcp", "bridge", "ontop")

AU_A = 4.08
VACUUM_MARGIN = 10.0       # ASE centers slab with this margin on each z side
FMAX = 0.003               # eV/Angstrom
MAX_STEPS = 500
MAXSTEP = 0.10             # Angstrom
SITE_TOL = 0.35            # approximate final-site label only
BRIDGE_OFFSET = 0.05       # Angstrom; only used if bridge survives

MODEL = "medium"           # explicit original MACE-MP medium checkpoint
DEVICE = "cpu"
DTYPE = "float64"
USE_D3 = True
D3_DAMPING = "bj"
D3_XC = "pbe"

OUT = Path("results") / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
STRUCTURES, LOGS, TRAJECTORIES = OUT / "structures", OUT / "logs", OUT / "trajectories"


def slab(size):
    atoms = fcc111("Au", size=size, a=AU_A, vacuum=VACUUM_MARGIN, periodic=False)
    if not np.array_equal(atoms.pbc, [True, True, False]):
        raise RuntimeError(f"Unexpected slab PBC: {atoms.pbc}")
    if len(atoms) != int(np.prod(size)):
        raise RuntimeError(f"Unexpected Au count: {len(atoms)} for {size}")
    return atoms


def constrain_bottom(atoms, n_fixed):
    """ASE surface tags descend from bottom=n_layers to top=1."""
    n_layers = atoms.get_tags().max()
    atoms.set_constraint(
        FixAtoms(mask=atoms.get_tags() > n_layers - n_fixed) if n_fixed else []
    )
    return atoms


def make_calc():
    return mace_mp(
        model=MODEL,
        device=DEVICE,
        default_dtype=DTYPE,
        dispersion=USE_D3,
        damping=D3_DAMPING,
        dispersion_xc=D3_XC,
    )


def save_atoms(atoms, stem):
    image = atoms.copy()
    image.info.pop("adsorbate_info", None)
    write(STRUCTURES / f"{stem}.vasp", image, format="vasp")
    write(STRUCTURES / f"{stem}.extxyz", image, format="extxyz")


def relax(atoms, calc, name):
    save_atoms(atoms, f"initial_{name}")
    atoms.calc = calc
    e_initial = float(atoms.get_potential_energy())
    opt = BFGS(
        atoms,
        trajectory=str(TRAJECTORIES / f"{name}.traj"),
        logfile=str(LOGS / f"{name}.log"),
        maxstep=MAXSTEP,
    )
    opt.run(fmax=FMAX, steps=MAX_STEPS)
    forces = atoms.get_forces()  # constraints are applied by ASE
    fmax = float(np.linalg.norm(forces, axis=1).max())
    save_atoms(atoms, f"relaxed_{name}")
    return {
        "E_initial": e_initial,
        "E_total": float(atoms.get_potential_energy()),
        "fmax": fmax,
        "steps": int(opt.nsteps),
        "converged": fmax <= FMAX,
    }


def xy_distance(atoms, point_a, point_b):
    delta = np.array([*(point_a - point_b), 0.0])
    mic, _ = find_mic(delta, atoms.cell, pbc=(True, True, False))
    return float(np.linalg.norm(mic[:2]))


def anchors_for(size):
    result = {}
    for site in SITES:
        ref = slab(size)
        add_adsorbate(ref, "Hg", height=2.0, position=site)
        result[site] = ref.positions[-1, :2].copy()
    return result


def final_site(atoms, anchors):
    hg_xy = atoms.positions[-1, :2]
    d = {s: xy_distance(atoms, hg_xy, p) for s, p in anchors.items()}
    site = min(d, key=d.get)
    return (site if d[site] <= SITE_TOL else "off-site"), d[site]


def add_case(cases, size, fixed, height, site, check):
    key = (size, fixed, height, site)
    cases.setdefault(key, set()).add(check)


def build_case_list():
    cases = {}
    # Lateral-cell convergence at 4 layers, all Au mobile, 2.0-A start.
    for size in SIZES:
        for site in SITES:
            add_case(cases, size, 0, 2.0, site, "cell_size")
    # Bottom-layer constraint sensitivity at the reference 4x4x4 cell.
    for fixed in FIXED_BOTTOM_LAYERS:
        for site in SITES:
            add_case(cases, (4, 4, 4), fixed, 2.0, site, "fixed_layers")
    # Initial-height multi-start at 4x4x4 with all Au mobile.
    for height in HG_HEIGHTS:
        for site in SITES:
            add_case(cases, (4, 4, 4), 0, height, site, "starting_height")
    return cases


def main():
    for folder in (OUT, STRUCTURES, LOGS, TRAJECTORIES):
        folder.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUT.resolve()}", flush=True)
    calc = make_calc()

    hg = Atoms("Hg", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=False)
    hg.calc = calc
    e_hg = float(hg.get_potential_energy())
    write(STRUCTURES / "isolated_Hg.extxyz", hg)

    cases = build_case_list()
    slab_refs, clean_slabs, rows, relaxed = {}, {}, [], {}
    anchors = {size: anchors_for(size) for size in SIZES}
    for size, fixed, height, site in cases:
        slab_key = (size, fixed)
        if slab_key not in slab_refs:
            label = f"slab_{size[0]}x{size[1]}x{size[2]}_fix{fixed}"
            clean = constrain_bottom(slab(size), fixed)
            ref = relax(clean, calc, label)
            slab_refs[slab_key] = (ref, label)
            clean_slabs[slab_key] = clean.copy()
            print(
                f"{label}: E={ref['E_total']:.7f}, "
                f"fmax={ref['fmax']:.4f}, converged={ref['converged']}"
            )

        ref, slab_label = slab_refs[slab_key]
        # Use the matching relaxed reference substrate and same constraints.
        clean = clean_slabs[slab_key].copy()
        clean.calc = None
        atoms = clean.copy()
        atoms.calc = None
        add_adsorbate(atoms, "Hg", height=height, position=site)
        check_names = "+".join(sorted(cases[(size, fixed, height, site)]))
        name = f"{size[0]}x{size[1]}x{size[2]}_fix{fixed}_h{height:.1f}_{site}"
        result = relax(atoms, calc, name)
        site_final, site_dist = final_site(atoms, anchors[size])
        row = {
            "checks": check_names,
            "size": "x".join(map(str, size)),
            "fixed_bottom_layers": fixed,
            "initial_height_A": height,
            "start_site": site,
            "final_site_approx": site_final,
            "distance_to_final_site_A": site_dist,
            "E_slab_eV": ref["E_total"],
            "E_Hg_eV": e_hg,
            "E_total_eV": result["E_total"],
            "E_ads_eV": result["E_total"] - ref["E_total"] - e_hg,
            "fmax_eV_A": result["fmax"],
            "steps": result["steps"],
            "converged": result["converged"] and ref["converged"],
        }
        rows.append(row)
        relaxed[(size, fixed, height, site)] = atoms
        print(
            f"{name}: Eads={row['E_ads_eV']:.6f} eV, "
            f"fmax={row['fmax_eV_A']:.4f}, "
            f"{row['final_site_approx']}, converged={row['converged']}"
        )

    # Small perturbations around the baseline bridge only if it actually remains there.
    bridge_key = ((4, 4, 4), 0, 2.0, "bridge")
    bridge_row = next(r for r in rows if r["size"] == "4x4x4" and
                      r["fixed_bottom_layers"] == 0 and r["initial_height_A"] == 2.0 and
                      r["start_site"] == "bridge")
    extra = []
    if bridge_row["final_site_approx"] == "bridge" and bridge_row["converged"]:
        base = relaxed[bridge_key]
        u = base.cell[0, :2] / np.linalg.norm(base.cell[0, :2])
        v = np.array([-u[1], u[0]])
        i_hg = base.get_chemical_symbols().index("Hg")
        ref = slab_refs[((4, 4, 4), 0)][0]
        for axis, direction in (("a", u), ("b", v)):
            for sign, suffix in ((1, "plus"), (-1, "minus")):
                atoms = base.copy()
                atoms.calc = None
                atoms.positions[i_hg, :2] += sign * BRIDGE_OFFSET * direction
                name = f"bridge_offset_{axis}_{suffix}"
                result = relax(atoms, calc, name)
                site_final, site_dist = final_site(atoms, anchors[(4, 4, 4)])
                extra.append({
                    "checks": "bridge_offset",
                    "size": "4x4x4",
                    "fixed_bottom_layers": 0,
                    "initial_height_A": 2.0,
                    "start_site": name,
                    "final_site_approx": site_final,
                    "distance_to_final_site_A": site_dist,
                    "E_slab_eV": ref["E_total"],
                    "E_Hg_eV": e_hg,
                    "E_total_eV": result["E_total"],
                    "E_ads_eV": result["E_total"] - ref["E_total"] - e_hg,
                    "fmax_eV_A": result["fmax"],
                    "steps": result["steps"],
                    "converged": result["converged"] and ref["converged"],
                })
    else:
        print("Bridge perturbation skipped: the baseline bridge start relaxed away.")
    rows.extend(extra)

    fields = list(rows[0])
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    settings = {
        "model": MODEL, "device": DEVICE, "dtype": DTYPE,
        "dispersion": USE_D3, "D3_damping": D3_DAMPING, "D3_XC": D3_XC,
        "Au_a_A": AU_A, "vacuum_margin_each_side_A": VACUUM_MARGIN,
        "PBC": [True, True, False], "FMAX_eV_A": FMAX,
        "max_steps": MAX_STEPS, "maxstep_A": MAXSTEP,
        "size_tests": SIZES, "fixed_bottom_layers_tests": FIXED_BOTTOM_LAYERS,
        "height_tests_A": HG_HEIGHTS,
        "note": "One-factor-at-a-time sensitivity suite; no parameter cross-product.",
    }
    for package in ("ase", "mace-torch", "torch-dftd", "torch"):
        try:
            settings[f"version_{package}"] = version(package)
        except PackageNotFoundError:
            settings[f"version_{package}"] = "unknown"
    (OUT / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    with (OUT / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Hg/Au(111) one-factor-at-a-time validation | MACE-MP medium + D3\n")
        f.write("E_ads = E_total - E_slab - E_Hg; negative means favorable.\n")
        f.write("Compare E_ads within each sensitivity family; never compare raw total energies across cell sizes.\n\n")
        f.write(f"{ 'Checks':<22}{'Size':<10}{'Fix':>4}{'H0':>6}{'Start':>11}{'Final':>12}{'E_ads':>13}{'fmax':>10}{'OK':>7}\n")
        for r in rows:
            f.write(
                f"{r['checks']:<22}{r['size']:<10}{r['fixed_bottom_layers']:>4d}"
                f"{r['initial_height_A']:>6.1f}{r['start_site']:>11}"
                f"{r['final_site_approx']:>12}{r['E_ads_eV']:>13.6f}"
                f"{r['fmax_eV_A']:>10.4f}{str(r['converged']):>7}\n"
            )
        f.write("\nApproximate final-site labels are geometric diagnostics, not proof of a minimum.\n")
        f.write("Bridge perturbations screen stability; they do not replace a Hessian or DFT validation.\n")
        f.write("Use extxyz/traj to preserve partial PBC; VASP is for viewing.\n")

    print(f"\nDone: {OUT.resolve()}")
    print("Up to 32 adsorption relaxations + 5 matching clean-slab references;")
    print("up to 4 bridge-offset relaxations only if the baseline bridge survives.")


if __name__ == "__main__":
    main()
