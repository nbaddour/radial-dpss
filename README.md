# Finite radial prolate framework: pipeline and CPSWF datasets

Version 1.0.0. This archive is the computational companion to the manuscript
*Fourier–Bessel Ritz Approximation of Circular Prolate Spheroidal Wave Functions* by
Natalie Baddour. It holds the code that produced every number and figure in that
manuscript, and the datasets those numbers were measured against.

**Author.** Natalie Baddour, Department of Mechanical Engineering, University of Ottawa,
Ottawa, Ontario, Canada K1N 6N5. ORCID
[0000-0002-7025-7501](https://orcid.org/0000-0002-7025-7501). Contact:
nbaddour@uottawa.ca.

**Funding.** Natural Sciences and Engineering Research Council of Canada (NSERC), Discovery
Grant RGPIN-2023-03392.

The organizing parameter is the space-bandwidth product `c = KR`, where `K` is the Hankel
bandwidth and `R` is the disc radius. `n` is the angular (Bessel) order and `P` is the
Fourier-Bessel trial dimension.

## Layout

```
code/        the pipeline, the Phase 3 experiments, the figure scripts, and tests/
data/        benchmark ground truth, campaign arrays, pilot matrices, Phase 3 artifacts
figures/     the publication figures, PNG and PDF
LICENSE      MIT, covering everything in code/
CITATION.cff citation metadata
RELEASE.md   how to reproduce
MANIFEST.md  file inventory with counts and sizes
```

Paths inside the scripts resolve relative to this folder. Each module computes its
repository root from its own location, so the tree works wherever it is unpacked. The
environment variable `RDPSS_REPO` overrides that if a script is run from outside the tree.

## Install and test

```bash
cd code
pip install -r requirements.txt
python -m pytest tests -q
```

The pinned baseline is numpy 2.2.6, scipy 1.15.3, mpmath 1.3.0, pytest 9.1.1 and
matplotlib 3.10.9. The test suite asserts the core numerical invariant of each pipeline
module: symmetry and positive semidefiniteness of the assembled matrix, eigensolver
residual and determinism, exactness of the synthesis isometry and of the nodal
interpolation, the origin limit, the weighted-orthogonality correction, the conditioning
monitor trigger, and continuity of the mode tracking. It runs in about a second.

The interval-certification track additionally needs `python-flint`, pinned in
`code/requirements_phase3_certified.txt`. Everything else runs without it.

## What is in code/

The pipeline proper is the Task 12 group. `task_12_1_assemble_matrix.py` builds the finite
radial prolate matrix from its closed-form entries by panel-Gauss quadrature and supplies
the Shannon number `M_n(n,c)`. `task_12_2_eigensolver.py` performs the symmetric
eigensolve with diagnostics. `task_12_3_canonicalize.py` fixes order, normalization and
sign. `task_12_4_synthesis.py` provides the exact synthesis and interpolation maps and the
four-source error split. `task_12_5_origin.py` stabilizes the reconstruction near `r = 0`.
`task_12_6_worthogonality.py` measures the weighted-orthogonality defect and applies the
Löwdin correction. `task_12_7_monitor.py` reports conditioning and triggers the Nyström
fallback. `task_12_8_tracking.py` follows the eigenvalue trajectories through a sweep in
`c`.

The reference solver and the ground-truth dataset builder are the Task 10 group; the
Gate-C pilot is Task 11. The Task 13 group is the validation campaign. The Task 17 group
draws the publication figures.

The Phase 3 experiments are the `task_25_4_phase3*` group. Each script writes the file its
name matches, into `data/task_25_4_phase3/`:

| script | writes |
|---|---|
| `task_25_4_phase3_freeze_manifest.py` | `reference_mode_manifest.json` and its `.sha256` |
| `task_25_4_phase3_gate0_interval.py` | `gate0_interval_stack.json` |
| `task_25_4_phase3_certification.py` | `track_A_certification.json`, `track_A_cases/` |
| `task_25_4_phase3_certification_audit.py` | `track_A_reference_audit.json` |
| `task_25_4_phase3_structure.py` | `track_B_structure.json` |
| `task_25_4_phase3_oracle.py` | `track_C_oracle.json` |
| `task_25_4_phase3_mechanism.py` | `track_C_fixed_offsets.json` |
| `task_25_4_phase3_boundary.py` | `track_D_boundary.json` |
| `task_25_4_phase3_timing.py` | `track_E_timing.json` |
| `task_25_4_phase3.py` | the CSV summaries, `manifest.json`, `artifact_inventory.json` |
| `task_25_4_phase3_boundary_figure.py` | `figures/phase3_boundary.png` and `.pdf` |

`task_25_4_phase3_interval_ritz.py` is the interval-arithmetic Ritz module the
certification track calls. `task_27_4_augmented_conditioning.py` produces
`code/task_27_4_conditioning.json`, the conditioning table of the augmented matrix.

The short `t1*`, `t2*`, `a*_check`, `c*_check`, `eps_*`, `dht_check` and `ref_check`
scripts are single-claim checks. Each one prints the quantity named in the manuscript
passage it belongs to.

## What is in data/

`benchmark/` holds the trusted CPSWF ground truth: sixteen `.npz` files over
`n = 0,1,2,4` and `c = 10,20,40,80`, with a manifest and a README describing the arrays.
It can be rebuilt with `task_10_4_build_dataset.py`.

`campaign/` holds the Task 13 validation arrays, one `.npz` per experiment.

`pilot/` holds the Gate-C pilot matrices and solutions from Task 11.

`task_25_4_phase3/` holds the Phase 3 artifacts. `reference_mode_manifest.json` is the
frozen reference spectrum every Phase 3 measurement is taken against, and its SHA-256 is
stored beside it in `reference_mode_manifest.json.sha256`. The `track_*` JSON files are the
full per-case records; the `.csv` files are the flattened summaries;
`artifact_inventory.json` records the SHA-256 of each one. `track_A_cases/` holds the
sixteen per-case interval-certification records, which are the largest files here at about
40 MB together. `figures/` holds the Phase 3 diagnostic plots.

`track_D_boundary.json` is the source of the boundary-augmentation results and of the
comparison with Boulsane's construction at matched dimension. It is the file the
manuscript's numbers came from. `track_D_boundary_REPRO.json` is a later re-run of the same
script, and `reproduction_check_track_D.md` records the comparison between the two.

## Licences

Code in `code/` is MIT (`LICENSE`). Datasets in `data/` are CC-BY-4.0
(`data/DATA_LICENSE.txt`). The figures follow the code licence.

## Citing

`CITATION.cff` carries this archive's citation metadata in machine-readable form.
