# Release and reproducibility guide

Version 1.0.0 of the archival artifact accompanying the manuscript *Fourier–Bessel Ritz
Approximation of Circular Prolate Spheroidal Wave Functions* by Natalie Baddour.

**Author.** Natalie Baddour, Department of Mechanical Engineering, University of Ottawa,
Ottawa, Ontario, Canada K1N 6N5. ORCID
[0000-0002-7025-7501](https://orcid.org/0000-0002-7025-7501). Contact:
nbaddour@uottawa.ca.

**Funding.** Natural Sciences and Engineering Research Council of Canada (NSERC), Discovery
Grant RGPIN-2023-03392.

This file describes what the archive contains and how to reproduce its results.

---

## 1. What the artifact is

A self-contained reproducible package: the finite radial prolate pipeline (`code/`), the
Phase 3 experiment scripts, the figure scripts, and the datasets (`data/`) that are the
ground truth, the campaign outputs, and the Phase 3 records behind the manuscript's
sections, tables and figures.

**Licences.** Source code under **MIT** (`LICENSE`); datasets under **CC-BY-4.0**
(`data/DATA_LICENSE.txt`). Citation metadata is in `CITATION.cff`.

`MANIFEST.md` lists what is here, folder by folder, with file counts and sizes.

---

## 2. Reproduce the results

### Install

```bash
cd code
pip install -r requirements.txt
```

Pinned baseline: numpy 2.2.6, scipy 1.15.3, mpmath 1.3.0, pytest 9.1.1, matplotlib 3.10.9.
The interval-certification track needs `python-flint` in addition, pinned in
`requirements_phase3_certified.txt`.

### Test suite

```bash
cd code
python -m pytest tests -q            # expect: 21 passed
```

The pilot- and benchmark-data tests are included in that count, since the datasets ship
with the code. Tests that need `data/pilot` or `data/benchmark` skip cleanly rather than
fail if those are removed.

### Regenerate the pipeline outputs

```bash
cd code
python task_10_4_build_dataset.py               # rebuild the benchmark ground truth (optional)
python task_13_7_steady_state_plot.py           # campaign arrays and the headline figure
python task_17_5_publication_figures.py         # the Task 17.5 figures
python task_17_6_shannon_number.py              # the Shannon-number figure
```

### Regenerate the Phase 3 artifacts

The Phase 3 measurements are taken against a frozen reference spectrum, so the order
matters. `reference_mode_manifest.json` is written first and every later script reads it.
Its SHA-256 is stored in `reference_mode_manifest.json.sha256`; check that before relying
on a re-run.

```bash
cd code
python task_25_4_phase3_freeze_manifest.py        # reference_mode_manifest.json + .sha256
python task_25_4_phase3_gate0_interval.py         # gate0_interval_stack.json
python task_25_4_phase3_certification.py          # track_A_certification.json, track_A_cases/
python task_25_4_phase3_certification_audit.py    # track_A_reference_audit.json
python task_25_4_phase3_structure.py              # track_B_structure.json
python task_25_4_phase3_oracle.py                 # track_C_oracle.json
python task_25_4_phase3_mechanism.py              # track_C_fixed_offsets.json
python task_25_4_phase3_boundary.py               # track_D_boundary.json
python task_25_4_phase3_timing.py                 # track_E_timing.json
python task_25_4_phase3.py                        # consolidate
python task_25_4_phase3_boundary_figure.py        # figures/phase3_boundary.{png,pdf}
```

`task_25_4_phase3.py` recomputes nothing. It reads the seven track files, then writes
`manifest.json`, the six CSV summaries (`modes`, `certification`, `structure`,
`mechanism`, `boundary`, `timing`), `reference.npz`, `failure_log.json`, the six
diagnostic plots in `data/task_25_4_phase3/figures/`, and `artifact_inventory.json`, which
records a SHA-256 for each of those outputs and for each of its seven inputs.

### Determinism

`scipy.linalg.eigh` is deterministic, and the eigensolver re-run test asserts bit-identical
eigenvalues. The only randomness is the synthesis round-trip, which uses a fixed seed
(`np.random.default_rng(0)`).

### Reproduction record for the boundary track

`task_25_4_phase3_boundary.py` was re-run from the SHA-256-verified manifest.
`data/task_25_4_phase3/track_D_boundary_REPRO.json` is that run's output and
`reproduction_check_track_D.md` records the comparison, over 16 cases and 128 rows. Every
figure quoted in the manuscript came out identical. The largest absolute difference anywhere
was 2.2e-14. The re-run used numpy 2.4.4 and scipy 1.17.1, a drift from the pinned 2.2.6
and 1.15.3, and that drift accounts for the sub-1e-14 differences.

`track_D_boundary.json` is the file the manuscript's numbers came from, under the pinned
versions, and it is the one that ships here.
