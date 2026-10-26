# Supporting Information

This repository contains input xyzs, output files, and plotting scripts used to generate figures and tables for the paper.

## Contents

- [input](input): Molecular input structures.
  - [input/water](input/water)
  - [input/gly](input/gly)
  - [input/misc](input/misc)
- [output](output): Raw program outputs organized as:
  - output/{code}/{basis}/{molecule}{size}.out
  - where code is one of: exess, gpu4pyscf, quick, qchem, accurate_sph, accurate_cart
  - [output/tight_convergence](output/tight_convergence): Outputs from tight SCF convergence runs, organized as:
    - output/tight_convergence/{code}/{molecule}{size}_{basis}.out
    - where code is one of: exess, gpu4pyscf, qchem, orca_reference
    - covering gly10 and water20 with cc-pVTZ and def2-TZVP
- [performance_output](performance_output): Performance plots, CSV, and LaTeX tables.
- [accuracy_output](accuracy_output): Accuracy plots, convergence behaviour plots, CSV (including MAD and RMSD), and LaTeX tables.
- [read_output.py](read_output.py): Shared parsing and plotting helper utilities.

## Reproducing Performance Outputs

From repository root:

```bash
python3 compare_performance.py
```

This writes outputs to [performance_output](performance_output), including:

- [performance_output/performance_timings.csv](performance_output/performance_timings.csv)
- [performance_output/exess_speedups_table.tex](performance_output/exess_speedups_table.tex)
- [performance_output/misc_timings_table.tex](performance_output/misc_timings_table.tex): GPU4PySCF and EXESS timings for the misc molecules
- PNGs named performance_{molecule}_{basis}.png

## Reproducing Accuracy Outputs

From repository root:

```bash
python3 compare_accuracy.py
```

This writes outputs to [accuracy_output](accuracy_output), including:

- [accuracy_output/energy_diffs.csv](accuracy_output/energy_diffs.csv)
- [accuracy_output/converged_energy_table.tex](accuracy_output/converged_energy_table.tex)
- [accuracy_output/tight_convergence_table.tex](accuracy_output/tight_convergence_table.tex): tight-convergence energies relative to the ORCA reference
- energy_diffs_{basis}_{molecule}.png
- per-system convergence plots in [accuracy_output/convergence_behaviour](accuracy_output/convergence_behaviour)