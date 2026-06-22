import os
import re
from pathlib import Path

import numpy as np
from matplotlib import ticker

METHODS = {
    "exess": {"color": "blue", "marker": "^", "linestyle": "--", "screening threshold": "$10^{-11}$", "label": "EXESS"},
    "gpu4pyscf": {
        "color": "red",
        "marker": "o",
        "linestyle": (0, (4, 2, 4, 2, 1, 2)),
        "screening threshold": "$10^{-10.5}$",
        "label": "GPU4PySCF",
    },
    "quick": {
        "color": "green",
        "marker": "D",
        "linestyle": (0, (3, 3, 1, 3, 1, 3)),
        "screening threshold": "$10^{-10}$",
        "label": "QUICK",
    },
    "qchem": {
        "color": "black",
        "marker": "X",
        "linestyle": "-.",
        "screening threshold": "$10^{-10}$",
        "label": "Q-Chem",
    },
}

FIGSIZE = (6, 3)
FONTSIZE = 12

NUM_BASIS_FUNCS = {
    "gly": {
        "6-31G*": (66, 19),
        "6-31G**": (75, 25),
        "cc-pVDZ": (75, 25),
        "cc-pVTZ": (185, 65),
        "cc-pVQZ": (385, 140),
        "def2-TZVP": (162, 48),
        "def2-QZVP": (393, 142),
    },
    "water": {
        "6-31G*": (19, 0),
        "6-31G**": (25, 0),
        "cc-pVDZ": (25, 0),
        "cc-pVTZ": (65, 0),
        "cc-pVQZ": (140, 0),
        "def2-TZVP": (48, 0),
        "def2-QZVP": (142, 0),
    },
}
BASIS_SETS = ["def2-TZVP", "cc-pVTZ", "def2-QZVP", "cc-pVQZ"]
MOLECULES = ["water", "gly"]
OUTPUT_ROOT = Path(__file__).resolve().parent / "output"


def collect_data(methods):
    data = {}
    for basis in BASIS_SETS:
        for molecule in MOLECULES:
            dir_path = OUTPUT_ROOT / "exess" / basis
            sizes = []
            for fname in os.listdir(dir_path):
                if fname.startswith(molecule) and fname.endswith(".out"):
                    size = int(fname[len(molecule) : -4].split("_")[0])
                    sizes.append(size)
            rows = []
            for size in sorted(sizes):
                entry = {"size": size}
                for code in methods:
                    file_path = OUTPUT_ROOT / code / basis / f"{molecule}{size}.out"
                    if file_path.exists():
                        entry[code] = file_path
                rows.append(entry)
            if rows:
                data[(basis, molecule)] = rows
    return data


def get_iter_data(file, line_pattern, average_over=range(1, 11, 1), start_from_pattern=None):
    try:
        with open(file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("can't find file", file)
        return []

    values = []
    wanted_hits = set(average_over) if average_over is not None else None
    pattern = re.compile(line_pattern)
    marker = re.compile(start_from_pattern) if start_from_pattern else None

    hit = 0
    for line in lines:
        if marker and marker.search(line):
            values = []
            hit = 0

        stripped = line.strip()
        if not stripped:
            continue
        for match in pattern.finditer(stripped):
            hit += 1
            if wanted_hits is not None and hit not in wanted_hits:
                continue
            val = float(match.group(1))
            if isinstance(val, (float, np.floating)) and np.isnan(val):
                continue
            values.append(val)

    return values


def get_accuracy_iter_data(file, line_pattern, start_from_pattern=None):
    return get_iter_data(
        file,
        line_pattern,
        average_over=range(1, 16),
        start_from_pattern=start_from_pattern,
    )


def parse_iteration_energies(path):
    path_parts = Path(path).parts
    output_idx = path_parts.index("output")
    code = path_parts[output_idx + 1]

    if code == "gpu4pyscf":
        pattern = r"^cycle=\s*\d+\s+E=\s*(-?\d+\.\d+)"
        energies = get_accuracy_iter_data(
            path,
            pattern,
            start_from_pattern=r"method = (RHF|UHF|ROHF)",
        )
    else:
        if code in ("exess", "accurate_sph", "accurate_cart"):
            pattern = r"^\s*\d+\s*\|\s*(-?\d+\.\d+)"
        elif code == "qchem":
            pattern = r"^\s*\d+\s+(-?\d+\.\d+)"
        else:
            assert code == "quick"
            pattern = r"^\|\s*\d+\s+(-?\d+\.\d+)"

        energies = get_accuracy_iter_data(
            path,
            pattern,
        )

    return energies


def get_time(file):
    path_parts = Path(file).parts
    output_idx = path_parts.index("output")
    code = path_parts[output_idx + 1]

    if code == "exess":
        fock_pattern = r"^\s*\d+\s*\|[^|]*\|[^|]*\|\s*([-+]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*\|"
        diis_pattern = r"^\s*\d+\s*\|[^|]*\|[^|]*\|[^|]*\|\s*([-+]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"
        fock_vals = get_iter_data(file, fock_pattern)
        diis_vals = get_iter_data(file, diis_pattern)
        if not fock_vals or not diis_vals:
            return np.nan
        return np.mean(fock_vals) + np.mean(diis_vals)
    else:
        if code == "quick":
            pattern = r"^\|\s*\d+\s+\S+\s+\S+\s+([-+]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"
        else:
            assert code == "gpu4pyscf"
            pattern = r"^CPU time for vj and vk on gpu\s+([-+]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+sec,\s+wall time"
        vals = get_iter_data(file, pattern)
        return np.mean(vals) if vals else np.nan


def set_pretty_axes(ax, sizes, basis, molecule, ylabel):
    slope, intercept = NUM_BASIS_FUNCS[molecule][basis]
    ax.set_xlabel(f"Number of basis functions ({basis})", fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(False)
    ax.tick_params(axis="y", which="minor", left=True, right=True)
    ax.xaxis.set_minor_locator(ticker.FixedLocator([]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{int(val)}"))
    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda x: (x - intercept) / slope, lambda x: x * slope + intercept),
    )
    prefix = "gly" if molecule == "gly" else r"$(H_2O)$"
    secax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{prefix}\\textsubscript{{{val:.0f}}}"))
    secax.xaxis.set_major_locator(ticker.FixedLocator(sizes[1::2]))
    secax.xaxis.set_minor_locator(ticker.FixedLocator(sizes))
    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelsize=FONTSIZE)
    plus = 1 if molecule == "gly" else 0
    secax.tick_params(axis="x", which="both", bottom=False, top=True, labelsize=FONTSIZE + plus)
