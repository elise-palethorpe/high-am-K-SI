import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from read_output import (
    FIGSIZE,
    FONTSIZE,
    METHODS,
    MOLECULES,
    NUM_BASIS_FUNCS,
    OUTPUT_ROOT,
    collect_data,
    get_iter_data,
    get_time,
    set_pretty_axes,
)

OUTDIR = Path(__file__).resolve().parent / "performance_output"
OUTDIR.mkdir(parents=True, exist_ok=True)

MISC_MOLECULES = ["GCGC", "buckycatcher", "taxol", "olestra", "morpholine"]

plt.style.use(["science"])
plt.rcParams["text.usetex"] = True


def annotate_speedup(slow, fast, sizes, basis, molecule, pos=0.06):
    speed_ups = [slow[i] / fast[i] for i in range(len(slow))]
    best_index = max(range(len(speed_ups)), key=lambda i: speed_ups[i])
    max_speed_up = slow[best_index] / fast[best_index]
    slope, intercept = NUM_BASIS_FUNCS[molecule][basis]
    bfs = [s * slope + intercept for s in sizes]
    x_range = max(bfs) - min(bfs)

    offset = 0.02 * x_range
    bar_length = 0.01 * x_range
    text_offset = pos * x_range
    plt.plot(
        [bfs[best_index] + offset, bfs[best_index] + offset],
        [slow[best_index], fast[best_index]],
        color="black",
        linestyle="-",
    )
    plt.hlines(
        slow[best_index], bfs[best_index] + offset - bar_length, bfs[best_index] + offset + bar_length, color="black"
    )
    plt.hlines(
        fast[best_index], bfs[best_index] + offset - bar_length, bfs[best_index] + offset + bar_length, color="black"
    )
    text_y = math.sqrt(slow[best_index] * fast[best_index]) * 0.9
    plt.text(
        bfs[best_index] + offset + text_offset,
        text_y,
        f"{max_speed_up:.1f}$\\times$",
        fontsize=FONTSIZE - 1,
        va="center",
    )


def plot_all(rows, annotate=True):
    for (basis, molecule), panel_rows in rows.items():
        fig, ax = plt.subplots(figsize=(FIGSIZE))
        slope, intercept = NUM_BASIS_FUNCS[molecule][basis]
        sizes = [row["size"] for row in panel_rows]
        xvals = [size * slope + intercept for size in sizes]
        times = {}
        for code in METHODS.keys():
            times[code] = []
            exists = False
            for row in panel_rows:
                if code not in row:
                    times[code].append(np.nan)
                else:
                    exists = True
                    times[code].append(get_time(row[code]))
            if exists:
                style = METHODS[code]
                ax.plot(
                    xvals,
                    times[code],
                    label=style["label"],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    fillstyle="none",
                )
        if annotate:
            annotate_speedup(times["gpu4pyscf"], times["exess"], sizes, basis, molecule)

        set_pretty_axes(ax, sizes, basis, molecule, ylabel="Fock build time (s)")
        if "T" in basis:
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(y_min, 6 * times["gpu4pyscf"][-1])
        ax.legend(fontsize=FONTSIZE - 1, loc="lower right")
        plt.savefig(OUTDIR / f"performance_{molecule}_{basis}.png", dpi=600)
        plt.close()


def misc_rows():
    rows = []
    for molecule in MISC_MOLECULES:
        for exess_file in sorted((OUTPUT_ROOT / "exess" / "misc").glob(f"{molecule}_*.out"), reverse=True):
            gpu4pyscf_file = OUTPUT_ROOT / "gpu4pyscf" / "misc" / exess_file.name
            rows.append(
                {
                    "basis": exess_file.stem.split("_")[1],
                    "molecule": molecule,
                    "nbasis": int(get_iter_data(gpu4pyscf_file, r"^number of NR cGTOs = (\d+)", average_over=None)[-1]),
                    "exess": get_time(exess_file),
                    "gpu4pyscf": get_time(gpu4pyscf_file),
                }
            )
    return rows


def misc_latex_table(rows):
    lines = [r"\begin{table}", r"\centering", r"\begin{tabular}{cc|c|cc}", r"\hline"]
    lines.append(r"Molecule & Basis & $N$ sph. basis functions & GPU4PySCF (s) & EXESS (s) \\ \hline")
    for molecule in MISC_MOLECULES:
        mol_rows = [r for r in rows if r["molecule"] == molecule]
        for i, row in enumerate(mol_rows):
            mol_cell = rf"\multirow{{{len(mol_rows)}}}{{*}}{{{molecule}}}" if i == 0 else ""
            lines.append(
                f"{mol_cell} & {row['basis']} & {row['nbasis']} & {row['gpu4pyscf']:.2f} & {row['exess']:.2f}" + r" \\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Average of 5 SCF iteration timings in seconds of \texttt{GPU4PySCF} and \texttt{EXESS}.}"
    )
    lines.append(r"\label{tab:misc_timings}")
    lines.append(r"\end{table}")
    with open(OUTDIR / "misc_timings_table.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def csv_times(rows, misc):
    columns = ["basis", "molecule", "size", "nbasis", "exess", "gpu4pyscf", "quick"]
    with open(OUTDIR / "performance_timings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for (basis, molecule), panel_rows in rows.items():
            row_dict = {"basis": basis, "molecule": molecule}
            for row in panel_rows:
                row_dict["size"] = row.get("size", "")
                for method in ["exess", "gpu4pyscf", "quick"]:
                    if method not in row:
                        row_dict[method] = ""
                        continue
                    row_dict[method] = get_time(row[method])
                writer.writerow(row_dict)
        writer.writerows(misc)


def latex_tables(data, basis_filter):
    basis_labels = []
    for basis, _ in data:
        if basis_filter in basis and basis not in basis_labels:
            basis_labels.append(basis)
    basis_labels.sort()

    def get_speedup(panel_rows, method, size):
        for r in panel_rows:
            if r.get("size") == size and method in r and "exess" in r:
                ref_time = get_time(r[method])
                exess_time = get_time(r["exess"])
                return f"{ref_time/exess_time:.2f}"
        return "-"

    method_count = 2 if basis_filter == "T" else 1
    lines = [r"\begin{table*}", r"{", r"\begin{tabular}{c" + "|cc" * method_count + "}"]
    header1 = [""]
    header2 = [""]
    header1.append(rf"\multicolumn{{{len(basis_labels)}}}{{|c}}{{GPU4PySCF}}")
    header2.extend([basis for basis in basis_labels])
    if basis_filter == "T":
        header1.append(rf"\multicolumn{{{len(basis_labels)}}}{{|c}}{{QUICK}}")
        header2.extend([basis for basis in basis_labels])
    lines.append(" & ".join(header1) + r" \\")
    lines.append(" & ".join(header2) + r" \\ \hline")
    for molecule in MOLECULES:
        sizes = [row["size"] for row in data[(basis_labels[0], molecule)]]
        mol = "(H2O)" if molecule == "water" else "gly"
        for size in sizes:
            row = [rf"\ce{{{mol}_{size}}}"]
            for method in ["gpu4pyscf", "quick"]:
                if basis_filter == "T" or method != "quick":
                    for basis in basis_labels:
                        panel_rows = data[(basis, molecule)]
                        row.append(get_speedup(panel_rows, method, size))
            lines.append(" & ".join(row) + r" \\")
    lines.append(r"\end{tabular}")
    basis_str = "triple" if "T" in basis_filter else "quadruple"
    quick_str = r"and \texttt{QUICK} " if basis_filter == "T" else ""
    com = "," if basis_filter == "T" else " and"
    quick_thresh_str = r" and $10^{-10}$" if basis_filter == "T" else ""
    lines.append(
        rf"\caption{{Single H200 GPU Fock build speedups of the hybrid LA-HA scheme implemented in \texttt{{EXESS}} over \texttt{{GPU4PySCF}} {quick_str}with {basis_str}-$\zeta$ basis sets. Timings are averaged over the first five iterations of an incremental Fock build and have been tested with screening thresholds of $10^{{-11}}${com} $10^{{-10.5}}${quick_thresh_str} for \texttt{{EXESS}}{com} \texttt{{GPU4PySCF}} {quick_str}respectively.}}"
    )
    lines.append(rf"\label{{tab:{basis_filter}_speedups}}")
    lines.append(r"}")
    lines.append(r"\end{table*}")

    return lines


def write_latex_tables(data):
    with open(OUTDIR / "exess_speedups_table.tex", "w") as f:
        for basis_filter in ["T", "Q"]:
            lines = latex_tables(data, basis_filter)
            for line in lines:
                f.write(line + "\n")
            f.write("\n\n")


def main():
    data = collect_data(methods=["exess", "gpu4pyscf", "quick"])
    misc = misc_rows()
    csv_times(data, misc)
    write_latex_tables(data)
    misc_latex_table(misc)
    plot_all(data)


if __name__ == "__main__":
    main()
