# Manifest

Version 1.0.0. Totals: **227 files, about 52 MB**.

| Folder | Files | Size | Contents |
|---|---:|---:|---|
| `code/` | 98 | 0.55 MB | pipeline modules, Phase 3 scripts, figure scripts, single-claim checks, `tests/`, two requirements files, `README.md`, `t195_data.npz`, `task_27_4_conditioning.json` |
| `data/benchmark/` | 18 | 1.79 MB | 16 CPSWF ground-truth `.npz` over `n = 0,1,2,4` and `c = 10,20,40,80`, plus `manifest.json` and `README.md` |
| `data/campaign/` | 10 | 0.03 MB | Task 13 validation-campaign arrays, one `.npz` per experiment |
| `data/pilot/` | 28 | 0.96 MB | Gate-C pilot matrices `Bmat_*.npz` and solutions `sol_*.npz` |
| `data/task_25_4_phase3/` | 24 | 6.38 MB | frozen reference manifest and its SHA-256, the seven `track_*` JSON records, six CSV summaries, `reference.npz`, `artifact_inventory.json`, `failure_log.json`, two pilot interval records, the boundary re-run and its check |
| `data/task_25_4_phase3/track_A_cases/` | 16 | 40.12 MB | per-case interval-certification records, one per `(n,c)` |
| `data/task_25_4_phase3/figures/` | 6 | 0.46 MB | Phase 3 diagnostic plots |
| `figures/` | 19 | 1.89 MB | publication figures, PNG and PDF |
| root | 8 | 0.02 MB | `README.md`, `RELEASE.md`, `MANIFEST.md`, `LICENSE`, `CITATION.cff`, `.gitignore`, `.gitattributes`, and `data/DATA_LICENSE.txt` |

`track_A_cases/` is three quarters of the archive by size. It holds the evidence behind the
certification track, at the level of the individual certified interval, and is the only
part a reader can drop and still run everything else.

## Integrity

`data/task_25_4_phase3/artifact_inventory.json` carries a SHA-256 for each Phase 3
consolidation output and for each of the seven track files it was built from.
`data/task_25_4_phase3/reference_mode_manifest.json.sha256` carries the hash of the frozen
reference spectrum. `data/benchmark/manifest.json` describes the ground-truth arrays.

## Where the manuscript's numbers live

| Manuscript content | File |
|---|---|
| CPSWF ground truth used throughout | `data/benchmark/` |
| Validation campaign | `data/campaign/` |
| Interval certification | `data/task_25_4_phase3/track_A_certification.json`, `track_A_cases/`, `certification.csv` |
| Block monotonicity | `data/task_25_4_phase3/track_B_structure.json`, `structure.csv` |
| Offset mechanism and the oracle diagnostic | `data/task_25_4_phase3/track_C_fixed_offsets.json`, `track_C_oracle.json`, `mechanism.csv` |
| Boundary augmentation, and the comparison with Boulsane at matched dimension | `data/task_25_4_phase3/track_D_boundary.json`, `boundary.csv` |
| Assembly and eigensolve timing | `data/task_25_4_phase3/track_E_timing.json`, `timing.csv` |
| Conditioning of the augmented matrix | `code/task_27_4_conditioning.json` |
