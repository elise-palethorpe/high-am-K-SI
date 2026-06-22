import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib import ticker

from read_output import (
    BASIS_SETS,
    FIGSIZE,
    FONTSIZE,
    METHODS,
    MOLECULES,
    NUM_BASIS_FUNCS,
    collect_data,
    parse_iteration_energies,
    set_pretty_axes,
)

REFERENCE_METHODS = {
    "accurate_sph": {
        "label": "\makecell{Spherical \\\\ reference}",
        "screening threshold": "$10^{-20}$",
    },
    "accurate_cart": {
        "label": "\makecell{Cartesian \\\\ reference}",
        "screening threshold": "$10^{-20}$",
    },
}

ALL_METHODS = {**METHODS, **REFERENCE_METHODS}

OUTDIR = Path(__file__).resolve().parent / "accuracy_output"
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.style.use(["science"])
plt.rcParams["text.usetex"] = True


def get_method_energy_diff(row, method):
    ref_method = "accurate_cart" if method == "quick" else "accurate_sph"
    if method not in row or ref_method not in row:
        return None
    method_energies = parse_iteration_energies(row[method])
    ref_energies = parse_iteration_energies(row[ref_method])
    if not method_energies or not ref_energies:
        return None
    min_method = min(method_energies[1:])
    diff = abs(min_method - ref_energies[-1])
    return diff


def compute_mad_and_rmsd(data):
    results = {}
    for method in METHODS.keys():
        diffs = []
        for (_basis, _molecule), panel_rows in data.items():
            for row in panel_rows:
                diff = get_method_energy_diff(row, method)
                if diff is not None:
                    diffs.append(diff)
        if diffs:
            mad = np.mean(np.abs(diffs))
            rmsd = np.sqrt(np.mean(np.array(diffs) ** 2))
            results[method] = (mad, rmsd)
    return results


def csv_energy_diffs(rows, stats=None):
    columns = ["basis", "molecule", "size", "exess", "gpu4pyscf", "quick", "qchem"]
    with open(OUTDIR / "energy_diffs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for (basis, molecule), panel_rows in rows.items():
            row_dict = {"basis": basis, "molecule": molecule}
            for row in panel_rows:
                row_dict["size"] = row.get("size", "")
                for method in METHODS.keys():
                    diff = get_method_energy_diff(row, method)
                    row_dict[method] = diff if diff is not None else ""
                writer.writerow(row_dict)

        if stats:
            mad_row = {"basis": "MAD", "molecule": "", "size": ""}
            rmsd_row = {"basis": "RMSD", "molecule": "", "size": ""}
            for method in METHODS.keys():
                if method in stats:
                    mad, rmsd = stats[method]
                    mad_row[method] = mad
                    rmsd_row[method] = rmsd
                else:
                    mad_row[method] = ""
                    rmsd_row[method] = ""
            writer.writerow(mad_row)
            writer.writerow(rmsd_row)


def plot_energy_diffs(rows):
    for (basis, molecule), panel_rows in rows.items():
        fig, ax = plt.subplots(figsize=FIGSIZE)
        slope, intercept = NUM_BASIS_FUNCS[molecule][basis]
        sizes = [row["size"] for row in panel_rows]
        xvals = [size * slope + intercept for size in sizes]
        for method in METHODS.keys():
            diffs = []
            exists = False
            for row in panel_rows:
                diff = get_method_energy_diff(row, method)
                diffs.append(diff if diff is not None else np.nan)
                exists = exists or (diff is not None)
            if exists:
                style = METHODS[method]
                label = f"{style['label']} ({style['screening threshold']})"
                ax.plot(
                    xvals,
                    diffs,
                    label=label,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    fillstyle="none",
                )
        set_pretty_axes(ax, sizes, basis, molecule, ylabel=r"$|E_{\mathrm{min}} - E_{\mathrm{ref}}|$ ($E_h$)")
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 10)
        ax.legend(fontsize=FONTSIZE - 1, loc="upper left", ncol=2, columnspacing=1.2, handletextpad=0.6)
        fig.savefig(OUTDIR / f"energy_diffs_{basis}_{molecule}.png", dpi=600)
        plt.close(fig)


def plot_iteration_energy_diffs(rows):
    conv_dir = OUTDIR / "convergence_behaviour"
    conv_dir.mkdir(parents=True, exist_ok=True)
    for (basis, molecule), panel_rows in rows.items():
        for row in panel_rows:
            size = row["size"]
            fig, ax = plt.subplots(figsize=FIGSIZE)
            for method in METHODS.keys():
                if method not in row:
                    continue
                energies = parse_iteration_energies(row[method])
                ref_method = "accurate_cart" if method == "quick" else "accurate_sph"
                ref_energies = parse_iteration_energies(row[ref_method])
                if not energies or not ref_energies:
                    continue
                iters = list(range(1, len(energies) + 1))
                diffs = [abs(e - ref_energies[-1]) for e in energies]
                style = METHODS[method]
                label = f"{style['label']} ({style['screening threshold']})"
                ax.plot(
                    iters,
                    diffs,
                    label=label,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    fillstyle="none",
                )
                min_idx = np.argmin(energies[1:]) + 1
                ax.plot(
                    iters[min_idx],
                    diffs[min_idx],
                    marker=style["marker"],
                    color=style["color"],
                    linestyle="None",
                    fillstyle="full",
                    zorder=10,
                )
            ax.set_xlabel("Iteration number", fontsize=FONTSIZE)
            ax.set_ylabel(r"$|E_{\mathrm{iter}} - E_{\mathrm{ref}}|$ ($E_h$)", fontsize=FONTSIZE)
            ax.set_yscale("log")
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=range(2, 10)))
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            ax.grid(False)
            ax.tick_params(axis="x", which="minor", bottom=False, top=False)
            ax.legend(fontsize=FONTSIZE - 1)
            fig.savefig(conv_dir / f"{basis}_{molecule}{size}_iteration_energy_diffs.png", dpi=600)
            plt.close(fig)


def latex_table(data, basis):
    def get_converged_energy(row, method):
        if method not in row:
            return None
        method_energies = parse_iteration_energies(row[method])
        if "accurate" in method:
            return method_energies[-1] if method_energies else None
        return min(method_energies[1:]) if method_energies else None

    method_count = 6 if "T" in basis else 4
    lines = [r"\begin{table*}{", "\centering"]
    if method_count == 6:
        lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{c" + "|c" * method_count + "}")
    header1 = [""]
    header2 = [""]
    for method in ALL_METHODS.keys():
        if (method == "quick" or method == "accurate_cart") and "Q" in basis:
            continue
        method_label = ALL_METHODS[method]["label"]
        if method == "accurate_sph" and "Q" in basis:
            method_label = r"Reference"
        screening_threshold = ALL_METHODS[method]["screening threshold"]
        header1.append(rf"{method_label}")
        header2.append(rf"{screening_threshold}")
    lines.append(" & ".join(header1) + r" \\")
    lines.append(" & ".join(header2) + r" \\ \hline")
    for molecule in MOLECULES:
        mol = "(H2O)" if molecule == "water" else "gly"
        for method_info in data[(basis, molecule)]:
            size = method_info["size"]
            row = [rf"\ce{{{mol}_{size}}}"]
            for method in ALL_METHODS.keys():
                if (method == "quick" or method == "accurate_cart") and "Q" in basis:
                    continue
                energy = get_converged_energy(method_info, method)
                row.append(f"{energy:.8f}" if energy is not None else "-")
            lines.append(" & ".join(row) + r" \\")
    lines.append(r"\end{tabular}")
    if method_count == 6:
        lines.append(r"}")
    quick_str = r"\texttt{QUICK}, " if "T" in basis else ""
    quick_comment_str = (
        r"\texttt{QUICK} energies are evaluated in the Cartesian basis so a Cartesian reference energy has also been provided. "
        if "T" in basis
        else ""
    )
    caption = (
        r"\caption{Converged energies from \texttt{EXESS}, \texttt{GPU4PySCF}, "
        rf"{quick_str}"
        r"\texttt{{Q-Chem}} "
        rf"for a variety of systems with the {basis} basis set reported to 8dp. "
        r"The reference energies are evaluated using \texttt{EXESS} with a screening threshold of $10^{-20}$ "
        r"and energy convergence threshold of $10^{-10}$. "
        rf"{quick_comment_str}"
        r"Converged energies are the minimum energy obtained in 15 SCF iterations of an incremental Fock build.}"
    )
    lines.append(caption)
    lines.append(rf"\label{{tab:{basis}_accuracy}}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    return lines


def write_all_latex_tables(data):
    with open(OUTDIR / "converged_energy_table.tex", "w") as f:
        for basis in BASIS_SETS:
            lines = latex_table(data, basis)
            for line in lines:
                f.write(line + "\n")
            f.write("\n\n")


def main():
    data = collect_data(methods=ALL_METHODS.keys())
    stats = compute_mad_and_rmsd(data)
    csv_energy_diffs(data, stats)
    write_all_latex_tables(data)
    plot_energy_diffs(data)
    plot_iteration_energy_diffs(data)


if __name__ == "__main__":
    main()
